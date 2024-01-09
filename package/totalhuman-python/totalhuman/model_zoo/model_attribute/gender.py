import os
import numpy as np

import os
import cv2
import time
import torch

from ..model_common import load_onnx, load_tensorRT, load_openvino, load_tensorRT_multiple
from ...utils.utils_attribute.gender_util import normalization, preprocessing
from ...data.image import read_image

# from model_zoo.model_common import load_onnx, load_tensorRT, load_openvino, load_tensorRT_multiple
# from utils.utils_attribute.gender_util import normalization, preprocessing
# from data.image import read_image


class Attrb_Gender:
    def __init__(self,model_type,model_path,**kwargs):

        self.model_path = model_path
        self.model_type = model_type
        self.load_multi = kwargs.get("load_multi",False)

        self.img_size = kwargs.get("img_size",(256,128)) # h,w
        self.labels = ['female','male']

        if model_type in ['vino','openvino']:
            self.model_type = 'openvino'
            self.model_name = model_path[0].split("/")[-1]
        else:
            self.model_name = model_path.split("/")[-1]

        # model load
        if self.model_type=='onnx':
            onnx_device=kwargs.get("onnx_device",'cuda')
            output_sort=kwargs.get("output_sort",True)
            self.net = load_onnx.Onnx(self.model_path,output_sort=output_sort,onnx_device=onnx_device)
        elif self.model_type=='trt':
            if self.load_multi:
                torch.cuda.initialized = True
                torch.cuda.is_available()
                torch.cuda.set_device(0)
                self.net = load_tensorRT_multiple.load(self.model_path)
            else:
                self.net = load_tensorRT.TrtModel(self.model_path)
        elif self.model_type=='openvino':
            if self.load_multi:
                self.net = load_openvino.Openvino_multi(self.model_path)
            else:
                vino_device=kwargs.get('vino_device','CPU')
                self.net = load_openvino.Openvino(self.model_path,device=vino_device)

    # input type
    # numpy array  / norm / (N,C,H,W) - onnx, vino-multiple, tensorRT
    # torch tensor / norm / (N,C,H,W) - vino
    # torch tensor / norm / (N,C,H,W) / input.cuda() - tensorRT-multiple
    def infer_image(self,img): # input RGB image (crop human)
        img = preprocessing(img,self.img_size)

        out = self.net(img)[0]
        pred = np.argmax(out)
        pred_lb = self.labels[pred]

        return out,pred,pred_lb
