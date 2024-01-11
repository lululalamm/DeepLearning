import os
import numpy as np
import skimage.transform

import os
import os.path as osp
import cv2
import time
import torch

from ..model_common import load_onnx, load_openvino
from ...data.image import read_torchImage,resize_image_multi

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Expression_DAN:
    def __init__(self, model_type,model_path,**kwargs):
        
        self.label_num = kwargs.get('label_num',5)
        if self.label_num==5:
            self.exp_cls = ['happy','surprise','anger','sorrow','neurality']
        elif self.label_num==7:
            self.exp_cls = ["happy","embarrassed","anger","anxious","hurt","sorrow","neutrality"]

        self.model_path = model_path
        self.model_type=model_type
   
        self.load_multi = kwargs.get("load_multi",False)

        if self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,output_sort=True,onnx_device='cpu')
        
        elif self.model_type=='openvino':
            if self.load_multi:
                self.net = load_openvino.Openvino_multi(self.model_path,transform=False,output_sort=True)
                self.outs_len = len(self.net.output_names)
            else:
                self.net = load_openvino.Openvino(self.model_path,not_norm=True,torch_image=True,device='CPU')
                self.outs_len = self.net.outs_len

    def get(self,face,mask_off=False,eye_min=0): # RGB input image, 112x112 (aligned)

        if mask_off and np.argmax(face['mask_sf'])==1:
            face.expression=[]
            return []
        if eye_min>0 and 'eye_dist' in face.keys() and face.eye_dist<eye_min:
            face.expression=[]
            return []

        aimg = face['aimg']
        img = normalization(aimg)

        if self.model_type=='onnx':
            img = img.transpose(1,2,0)
        else:
            if self.model_type=='openvino' and self.load_multi:
                img = np.expand_dims(img,axis=0)
            else:
                img = torch.from_numpy(img).unsqueeze(0).float()


        output = self.net(img)[0]
        #output_sf = softmax(output)[0]
        pred = np.argmax(output)
        pred = self.exp_cls[pred]

        face.expression = pred

        return pred



