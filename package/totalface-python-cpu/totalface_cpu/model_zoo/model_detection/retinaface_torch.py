import numpy as np
import os
import cv2
import math
import torch
import time

from ..model_common import load_onnx, load_openvino
from ...data.image import resize_image,resize_image_multi

from ...utils.util_detection import PriorBox, decode, decode_torch, nms, decode_landm, decode_landm_torch


class RetinaFace:
    def __init__(self, model_type,model_path,**kwargs):
        self.model_path = model_path
        self.model_type = model_type
        self.model_name = 'retinaface_torch'
        
        self.nms_threshold = 0.4

        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.variance = [0.1, 0.2]
        self.clip = False
        self.load_multi = kwargs.get("load_multi",False)


        self.network = kwargs.get("network",'resnet50')

        if self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,onnx_device='cpu')
        elif self.model_type=='openvino':
            if self.load_multi:
                self.net = load_openvino.Openvino_multi(self.model_path,transform=False, output_sort=True)
            else:
                outputs_order = ['loc','conf','landms']
                self.net = load_openvino.Openvino(self.model_path,not_norm=True,torch_image=True,device='CPU',outputs_order=outputs_order)

        
    def forward(self,img,img_size):
        
        
        net_out_start=time.time()
        outs_ = self.net(img)
        net_out_end=time.time()


        if not outs_:
            return None

        
        priorbox = PriorBox(self.min_sizes, self.steps, self.clip, image_size=(img_size[0], img_size[1]),phase='val')

        priors = priorbox.vectorized_forward()
        prior_data = np.array(priors.data)

        if self.load_multi:
                
            outs=[]
            for o in outs_:
                outs.append(np.array(o))
                
            loc, conf, landms = outs # multiple
            loc = np.squeeze(loc,axis=0)
            conf = np.squeeze(conf,axis=0)
            landms = np.squeeze(landms,axis=0)
            
        else:
            outs = outs_

            loc, conf, landms = outs
            loc = np.reshape(loc,(prior_data.shape[0],-1))
            landms = np.reshape(landms,(prior_data.shape[0],-1))
            conf = np.reshape(conf,(prior_data.shape[0],-1))
    
        boxes = decode(loc, prior_data, self.variance)
        boxes = boxes * self.scale

        scores = conf[:,1]
        landms = decode_landm(landms, prior_data, self.variance)
        landms = landms * self.scale1

        net_out_time = (net_out_end-net_out_start)*1000
        
        return boxes, landms, scores, net_out_time
    
    def detect(self,img,thresh=0.3,input_size=(640,640),target_size=0,max_size=0,resize_method='resize'): # h,w


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
            if target_size==0:
                det_img = resize_image(img,(input_size[1],input_size[0]))
            else:
                det_img,_ = resize_image_multi(img,target_size,max_size)
                print("multi load:",det_img.shape)

            meta = {'original_shape':img.shape, 'resized_shape':det_img.shape}
            scale_x = meta['resized_shape'][1] / meta['original_shape'][1]
            scale_y = meta['resized_shape'][0] / meta['original_shape'][0]
                
        self.det_img = det_img
        self.det_shape = det_img.shape
        image_shape = det_img.shape[:2]

        det_img = np.float32(det_img)
        det_img -= (104, 117, 123)

        rescale_end = time.time()

        self.scale = np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        self.scale1 = np.array([image_shape[1],image_shape[0], image_shape[1], image_shape[0],
                               image_shape[1], image_shape[0], image_shape[1], image_shape[0],
                               image_shape[1], image_shape[0]])


        img_size = [det_img.shape[0],det_img.shape[1]]
        if not self.model_type=='onnx':
            det_img = det_img.transpose(2, 0, 1)
            if self.model_type=='openvino' and self.load_multi:
                det_img = np.expand_dims(det_img,axis=0)
            else:
                det_img = torch.from_numpy(det_img).unsqueeze(0)
        

        forward_start = time.time()
        outs = self.forward(det_img,img_size)
        forward_end = time.time()

        if not outs:
            return None
        
        boxes, landms, scores,net_out_time = outs

        post1_start = time.time()
        # ignore low scores
        inds = np.where(scores > thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        post1_end = time.time()

        rescale_time = (rescale_end-rescale_start)*1000
        forward_time = (forward_end-forward_start)*1000
        post1_time = (post1_end-post1_start)*1000
        time_dict={'rescale':rescale_time,"forward":forward_time,'post1':post1_time,'net_out':net_out_time}


        if resize_method=='resize':
            return dets, landms, scale_x, scale_y,time_dict
        else:
            return dets, landms,det_img, [pos_x, pos_y], det_scale,time_dict
