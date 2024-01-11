import os
import numpy as np
import skimage.transform

import os
import os.path as osp
import cv2
import time
import torch

from ...utils.util_detection import tensors_to_detections_np,weighted_non_max_suppression_np
from ..model_common import load_tensorRT, load_onnx, load_openvino,load_torch, load_tensorRT_multiple
from ...data.image import read_torchImage,resize_image_multi,resize_image

class BlazeFace:
    def __init__(self, model_type,model_path,isfront=False,**kwargs):
        self.model_path = model_path
        self.model_type=model_type
        self.isfront = isfront

        self.min_suppression_threshold = kwargs.get("iou_thresh",0.3)
        self.min_suppression_threshold = kwargs.get("nms_thresh",0.3)

        if model_type in ['vino','openvino']:
            self.model_base = os.path.join("/",*model_path[0].split("/")[:-1])
            self.model_name = model_path[0].split("/")[-1]
        else:
            self.model_base = os.path.join("/",*model_path.split("/")[:-1])
            self.model_name = model_path.split("/")[-1]
        
        if self.isfront:
            self.anchors_name = kwargs.get("anchors_name","blazeface_front_anchors.npy")
            self.anchors_path = os.path.join(self.model_base,self.anchors_name)
        else:
            self.anchors_name = kwargs.get("anchors_name","blazeface_back_anchors.npy")
            self.anchors_path = os.path.join(self.model_base,self.anchors_name)

        if self.model_type in ['pt','pth']:
            if self.isfront:
                self.net = load_torch.TorchModel('blazeface_front',self.model_path)
            else:
                self.net = load_torch.TorchModel('blazeface_back',self.model_path)
        elif self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,output_sort=True,onnx_device=kwargs.get("onnx_device",'cuda'))
        elif self.model_type=='trt':
                self.net = load_tensorRT.TrtModel(self.model_path,torch_image=True,not_norm=True)

        elif self.model_type=='openvino':
                self.net = load_openvino.Openvino(self.model_path,not_norm=True,torch_image=True,device=kwargs.get("device",'CPU'))

        self._init_vars()

    def _init_vars(self):

        self.num_anchors=896
        self.num_coords=16
        self.num_classes=1
        self.score_clipping_thresh = 100.0

        if self.isfront:
            #self.min_score_thresh = 0.75
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
        else:
            #self.min_score_thresh = 0.65
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
        
        self.anchors = np.array(np.load(self.anchors_path),dtype=np.float32)

    def forward(self,img,thresh,input_size):     

        net_out_start=time.time()
        outs = self.net(img)
        net_out_end=time.time()
        if self.model_type=='trt':
            for oi,o in enumerate(outs):
                if oi==0:
                    outs[oi] = np.reshape(outs[oi].ravel(),(-1,self.num_anchors,self.num_classes))
                else:
                    outs[oi] = np.reshape(outs[oi].ravel(),(-1,self.num_anchors,self.num_coords))


        # post processing
        detections = tensors_to_detections_np(outs[1],outs[0],self.anchors,self.num_anchors,self.num_coords,self.num_classes,self.score_clipping_thresh,thresh, \
                                                self.x_scale,self.y_scale,self.w_scale,self.h_scale)

        filtered_detections = []
        for i in range(len(detections)):
            faces = weighted_non_max_suppression_np(detections[i],self.min_suppression_threshold)
            faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, 17))
            filtered_detections.append(faces)

        net_out_time = (net_out_end-net_out_start)*1000
            
        return filtered_detections, net_out_time

    def detect(self,img,thresh=0.65,input_size = None,resize_method='pad'):

        if not input_size:
            if self.isfront:
                input_size=[128,128]
            else:
                input_size=[256,256]
        if resize_method=='pad':
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
            rescale_start = time.time()
            det_img = resize_image(img,(input_size[1],input_size[0]))
            meta = {'original_shape':img.shape, 'resized_shape':det_img.shape}
            scale_x = meta['resized_shape'][1] / meta['original_shape'][1]
            scale_y = meta['resized_shape'][0] / meta['original_shape'][0]

        self.det_shape = det_img.shape
        self.det_img = det_img

        # norm
        det_img = np.float32(det_img)
        det_img = (det_img / 127.5 ) - 1.0

        rescale_end = time.time()

        if not self.model_type=='onnx':
            det_img = det_img.transpose(2, 0, 1)
            det_img = torch.from_numpy(det_img).unsqueeze(0)

        forward_start = time.time()
        outs = self.forward(det_img,thresh,input_size)
        forward_end = time.time()

        if not outs:
            return None
        
        filtered_detections, net_out_time = outs

        post1_start = time.time()

        bboxs=[]
        keypoints=[]
        scores=[]

        for final in filtered_detections:
            for d in final:
                bbox = np.array([d[1],d[0],d[3],d[2]])
                bboxs.append(bbox)
                scores.append(d[16])

                eye_right = [d[4],d[5]]
                eye_left = [d[6],d[7]]
                nose = [d[8],d[9]]
                mouth = [d[10],d[11]]
                ear_right = [d[12],d[13]]
                ear_left = [d[14],d[15]]

                keypoints.append([eye_right, eye_left, nose,mouth, ear_right, ear_left])

        post1_end = time.time()

        rescale_time = (rescale_end-rescale_start)*1000
        forward_time = (forward_end-forward_start)*1000
        post1_time = (post1_end-post1_start)*1000

        time_dict={'rescale':rescale_time,"forward":forward_time,'post1':post1_time,'net_out':net_out_time}

        if resize_method=='resize':
            return np.array(bboxs), np.array(keypoints),np.array(scores), scale_x, scale_y,input_size,time_dict
        else:
            return np.array(bboxs), np.array(keypoints),np.array(scores), det_img, [pos_x, pos_y], det_scale,input_size,time_dict





