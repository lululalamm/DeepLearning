import os
import numpy as np
import skimage.transform

import os
import os.path as osp
import cv2
import time
import torch

from ...utils.util_detection import softmax, distance2bbox, distance2kps, nms, new_hw
from ..model_common import load_onnx, load_openvino
from ...data.image import resize_image_multi

class SCRFD_CV:
    def __init__(self, model_type,model_path,**kwargs):
        self.model_path = model_path
        self.model_type=model_type
        if model_type in ['vino','openvino']:
            self.model_name = model_path[0].split("/")[-1]
        else:
            self.model_name = model_path.split("/")[-1]
        
        self.center_cache = {}
        #self.nms_thresh = 0.4
        self.nms_thresh = kwargs.get("nms_thresh",0.4)
        #self.nms_thresh = kwargs.get("iou_thresh",0.4)

        self.load_multi = kwargs.get("load_multi",False)

        if self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,output_sort=True,onnx_device='cpu')
            self.outs_len = self.net.outs_len

        elif self.model_type=='openvino':
            if self.load_multi:
                self.net = load_openvino.Openvino_multi(self.model_path,transform=False,output_sort=True)
                self.outs_len = len(self.net.output_names)
            else:
                self.net = load_openvino.Openvino(self.model_path,not_norm=True,torch_image=True,device='CPU')
                self.outs_len = self.net.outs_len

        self._init_vars()


    def _init_vars(self):
        
        self.use_kps = False
        self._num_anchors = 1
        if self.outs_len==6:
            self.fmc = 2
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif self.outs_len==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif self.outs_len==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif self.outs_len==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self,img,thresh,input_size,multiscale=False):
        
        scores_list = []
        bboxes_list = []
        kpss_list = []

        input_height = input_size[0]
        input_width = input_size[1]

        net_out_start=time.time()
        outs = self.net(img)
        net_out_end=time.time()

        if outs[0].shape[0]==1:
            out_size=[outs[0].shape[1],outs[self.fmc].shape[1],outs[self.fmc*2].shape[1]]
            for i in range(0,self.outs_len-self.fmc+1,self.fmc):
                for j in range(0,self.fmc):
                    outs[i+j] = np.reshape(outs[i+j].ravel(),(out_size[i//self.fmc],-1)) 

        else:
            out_size=[outs[0].shape[0],outs[self.fmc].shape[0],outs[self.fmc*2].shape[0]]

        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx*fmc]
            bbox_preds = outs[idx*fmc+1]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = outs[idx*fmc+2] * stride

            height = round(input_height / stride)
            width = round(input_width / stride)
            height,width = new_hw(height,width,self._num_anchors,out_size[idx])

            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers
            pos_inds = np.where(scores>=thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        net_out_time = (net_out_end-net_out_start)*1000
            
        return scores_list, bboxes_list, kpss_list, net_out_time

    def detect(self,img,thresh=0.5, input_size = (640,640),target_size=0,max_size=0, max_num=0, metric='default'):
        if target_size==0:
            multiscale=False
            rescale_start = time.time()
            det_scale = 1.0
            det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8 )
            pos_y=0
            pos_x=0
            if img.shape[0] < input_size[0] and img.shape[1] < input_size[1]:
                pos_y=(input_size[0]-img.shape[0])//2
                pos_x=(input_size[1]-img.shape[1])//2
                det_img[pos_y:pos_y+img.shape[0], pos_x:pos_x+img.shape[1], :] = img
            elif img.shape[0]==img.shape[1] and img.shape[0]>input_size[0]:
                resize = input_size[0]//4*3
                det_scale = float(resize) / img.shape[0]
                img = cv2.resize(img, (resize,resize))
                pos_y=(input_size[0]-img.shape[0])//2
                pos_x=(input_size[1]-img.shape[1])//2
                det_img[pos_y:pos_y+img.shape[0], pos_x:pos_x+img.shape[1], :] = img
            else:
                im_ratio = float(img.shape[0]) / img.shape[1]
                model_ratio = float(input_size[1]) / input_size[0]
                if im_ratio>model_ratio:
                    new_height = input_size[1]
                    pos_y = 0
                    new_width = int(new_height / im_ratio)
                    pos_x = (input_size[0]-new_width)//2
                else:
                    new_width = input_size[0]
                    pos_x = 0
                    new_height = int(new_width * im_ratio)
                    pos_y = (input_size[1]-new_height)//2
                det_scale = float(new_height) / img.shape[0]
                resized_img = cv2.resize(img, (new_width, new_height))
                det_img[pos_y:pos_y+new_height, pos_x:pos_x+new_width, :] = resized_img

        else:
            multiscale=True
            rescale_start = time.time()
            det_img,det_scale = resize_image_multi(img,target_size,max_size)
            input_size = (det_img.shape[0],det_img.shape[1])
            print("multi load:",det_img.shape)

        self.det_shape = det_img.shape
        self.det_img = det_img
        
        # norm
        det_img = np.float32(det_img)
        det_img = (det_img - 127.5) / 128.0

        rescale_end = time.time()

        if not self.model_type=='onnx':
            det_img = det_img.transpose(2, 0, 1)
            if self.model_type=='openvino' and self.load_multi:
                det_img = np.expand_dims(det_img,axis=0)
            else:
                det_img = torch.from_numpy(det_img).unsqueeze(0)
 
        forward_start = time.time()
        outs = self.forward(det_img, thresh,input_size,multiscale)
        forward_end = time.time()

        if not outs:
            return None
        
        scores_list, bboxes_list, kpss_list, net_out_time = outs

        post1_start = time.time()
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        if self.use_kps:
            kpss = np.vstack(kpss_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det,self.nms_thresh)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        post1_end = time.time()

        rescale_time = (rescale_end-rescale_start)*1000
        forward_time = (forward_end-forward_start)*1000
        post1_time = (post1_end-post1_start)*1000

        time_dict={'rescale':rescale_time,"forward":forward_time,'post1':post1_time,'net_out':net_out_time}

        if target_size==0:
            return det, kpss, det_img, [pos_x, pos_y], det_scale,time_dict
        else:
            return det,kpss,det_img,det_scale,time_dict




