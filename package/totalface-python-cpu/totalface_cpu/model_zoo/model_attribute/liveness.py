import os
import numpy as np
import skimage.transform

import os
import os.path as osp
import cv2
import time
import torch

from ..model_common import load_onnx, load_openvino
from ...utils.util_attribute import CropImage

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Liveness:
    def __init__(self, model_type,model_path,**kwargs):

        self.liveness_lb={0:'real',1:'fake'}
        self.model_path = model_path
        self.model_type=model_type

        self.image_cropper = CropImage()
        self.scale = kwargs.get("scale",2.7)
        self.out_h = kwargs.get("out_h",80)
        self.out_w = kwargs.get("out_w",80)

        if model_type in ['vino','openvino']:
            self.model_name = model_path[0].split("/")[-1]
        else:
            self.model_name = model_path.split("/")[-1]
        
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

    def liveness_check(self,img,face,mask_off=False,eye_min=0,to_BGR=True): # BGR input original image, (x1,y1,x2,y2) bbox_ori

        if to_BGR:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
        if mask_off and np.argmax(face['mask_sf'])==1:
            face.liveness_sf=[]
            return []
        if eye_min>0 and 'eye_dist' in face.keys() and face.eye_dist<eye_min:
            face.liveness_sf=[]
            return []

        bbox_ori = face.bbox
        bbox = [int(bbox_ori[0]), int(bbox_ori[1]), int(bbox_ori[2]-bbox_ori[0]+1), int(bbox_ori[3]-bbox_ori[1]+1)]

        self.param = {
            "org_img": img,
            "bbox": bbox, # x1 y1 w h
            "scale": self.scale,
            "out_w": self.out_w,
            "out_h": self.out_h,
            "crop": True,
        }

        # preprocessing
        img = self.image_cropper.crop(**self.param)

        if not self.model_type=='onnx':
            if self.model_type=='openvino' and self.load_multi:
                img = np.expand_dims(img,axis=0)
            else:
                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img).unsqueeze(0).float()

        prediction,ft_map = self.net(img)
        output_sf = softmax(prediction)

        face.liveness_sf = output_sf

        return output_sf



