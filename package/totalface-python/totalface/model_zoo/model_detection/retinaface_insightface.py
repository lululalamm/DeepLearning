import numpy as np
import os
import cv2
import math
import torch

from ..model_common import load_tensorRT, load_onnx, load_openvino,load_torch
from ...data.image import read_image,resize_image

from ...utils.util_detection import generate_anchors_fpn, anchors_plane, _clip_pad, clip_boxes, nms, bbox_vote, bbox_pred,landmark_pred

class RetinaFace:
    def __init__(self,model_type,model_path,**kwargs):
        self.model_path = model_path
        self.model_type = model_type
        
        self.nms_threshold = 0.4
        self.default_size=640
        
        self.vote = False
        self.nocrop = False
        
        self._ratio = (1.,)
        self.anchor_cfg = None
        
        self.dense_anchor = False
        
        self.bbox_stds=[1.0,1.0,1.0,1.0]

        if self.model_type in ['pt','pth']:
            self.net = load_torch.TorchModel('retinaface_insightface',self.model_path)
        elif self.model_type=='onnx':
            #self.net = load_onnx.Onnx_cv(self.model_path,input_mean=0.0, input_std=1.0)
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,onnx_device=kwargs.get("onnx_device",'cuda'))
        elif self.model_type=='trt':
            self.net = load_tensorRT.TrtModel(self.model_path,not_norm=True,torch_image=True)
            self.shape = self.net.engine.get_binding_shape(0)
        elif self.model_type=='openvino':
            outputs_order = ['face_rpn_cls_prob_reshape_stride32',
                                'face_rpn_bbox_pred_stride32',
                                'face_rpn_landmark_pred_stride32',
                                'face_rpn_cls_prob_reshape_stride16',
                                'face_rpn_bbox_pred_stride16',
                                'face_rpn_landmark_pred_stride16',
                                'face_rpn_cls_prob_reshape_stride8',
                                'face_rpn_bbox_pred_stride8',
                                'face_rpn_landmark_pred_stride8']
            self.net = load_openvino.Openvino(self.model_path,not_norm=True,torch_image=True,device=kwargs.get("device"),outputs_order=outputs_order)

        self._init_vars() 
        
    def _init_vars(self):
        self.use_landmarks = True

        self.fpn_keys = []
        self._feat_stride_fpn = [32, 16, 8]
        
        self.anchor_cfg = {
                    '32' : {'SCALES' : (32, 16), 'BASE_SIZE' : 16, 'RATIOS': self._ratio, 'ALLOWED_BORDER': 9999},
                  '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': self._ratio, 'ALLOWED_BORDER': 9999},
                  '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': self._ratio, 'ALLOWED_BORDER': 9999},
                }
        
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)
            
        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=self.dense_anchor, cfg=self.anchor_cfg)))
        
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
            
        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

        self.re_nums=[4,8,20]
        
    def forward(self,img,thresh,input_size):
        
        proposals_list = []
        scores_list = []
        landmarks_list = []
        
        output = self.net(img)

        input_scale = input_size[1]/input_size[0]
        im_info = [input_size[0],input_size[1]]
        
        reshape_list=[]
        if self.model_type=='trt':
            for bind in self.net.engine:
                if "data" in bind: continue
                reshape_list.append(self.net.engine.get_binding_shape(bind))

            for oi,o in enumerate(output):
                output[oi] = output[oi].reshape(reshape_list[oi])
        
        for idx, s in enumerate(self._feat_stride_fpn):
            key = 'stride%s'%s
            stride = int(s)

            if self.use_landmarks:
                idx = idx*3
            else:
                idx = idx*2
            
            scores = output[idx]
            scores = scores[: , self._num_anchors['stride%s'%s]:, :, :]

            idx += 1
            bbox_deltas = output[idx]
   

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            A = self._num_anchors['stride%s'%s]
            K = height * width

            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            '''
            bbox_deltas[:,0::4] = bbox_deltas[:, 0::4] * self.bbox_stds[0]
            bbox_deltas[:,1::4] = bbox_deltas[:, 1::4] * self.bbox_stds[1]
            bbox_deltas[:,2::4] = bbox_deltas[:, 2::4] * self.bbox_stds[2]
            bbox_deltas[:,3::4] = bbox_deltas[:, 3::4] * self.bbox_stds[3]
            '''

            proposals = bbox_pred(anchors, bbox_deltas)
            #proposals = clip_boxes(proposals, im_info[:2])

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=thresh)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals_list.append(proposals)
            scores_list.append(scores)

            if not self.vote and self.use_landmarks:
                idx+=1
                #landmark_deltas = output[idx].cpu().detach().numpy()
                landmark_deltas = output[idx]
                landmark_deltas = _clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1]//A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                #landmarks[:,:,0:2] /= im_scale
                landmarks_list.append(landmarks)
                
        return scores_list, proposals_list, landmarks_list
        
        
        
    def detect(self,img,thresh=0.3,input_size=(640,640),target_size=0,max_size=0):

        re_image = resize_image(img,(input_size[1],input_size[0]))
        meta = {'original_shape':img.shape, 'resized_shape':re_image.shape}
        scale_x = meta['resized_shape'][1] / meta['original_shape'][1]
        scale_y = meta['resized_shape'][0] / meta['original_shape'][0]

        if not self.model_type in ['pt','pth','onnx']:
            re_image = re_image.transpose(2, 0, 1)
            re_image = torch.from_numpy(re_image).unsqueeze(0)

        scores_list, proposals_list, landmarks_list = self.forward(re_image,thresh,input_size)
        
        proposals = np.vstack(proposals_list)
        landmarks = None

        if proposals.shape[0]==0:
            if self.use_landmarks:
                landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks, scale_x, scale_y

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]

        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

        if not self.vote:
            keep = nms(pre_det,self.nms_threshold)
            det = np.hstack( (pre_det, proposals[:,4:]) )
            det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
        else:
            det = np.hstack( (pre_det, proposals[:,4:]) )
            det = bbox_vote(det,self.nms_threshold)
            
        return det, landmarks, scale_x, scale_y
        
        