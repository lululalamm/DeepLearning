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

class MaskDetector:
    def __init__(self, model_type,model_path,**kwargs):

        self.mask_lb={0:'unmask',1:'mask'}
        self.model_path = model_path
        self.model_type=model_type
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

    def mask_check(self,face): # RGB input image, 112x112 (aligned)

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
        # np.argmax(face.mask_sf)==1 -> mask
        # [unmask,mask]
        output_sf = softmax(output)[0] 

        face.mask_sf = output_sf
        return output_sf



