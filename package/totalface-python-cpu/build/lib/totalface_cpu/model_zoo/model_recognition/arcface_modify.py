from collections import OrderedDict
from torchvision import transforms as T
import torch.nn as nn
import torch

import cv2
import numpy as np

from ...data.image import read_torchImage, read_image
from ...utils.util_warp import face_align
from ...data.constant import LMARK_REF_ARC

from ..model_common import load_onnx, load_openvino
from ...utils.util_common import torch2numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Arcface:
    def __init__(self,model_type,model_path,**kwargs):#,out_size=112,num_features=512,network='r50',fp16=False):

        self.out_size=kwargs.get("out_size",112)
        self.num_features = kwargs.get("num_features",512)
        self.fp16=kwargs.get("fp16",False)
        self.model_path = model_path
        self.network = kwargs.get("network",'r50')
        self.load_multi = kwargs.get('load_multi',False)

        self.model_type = model_type

        self.not_norm = kwargs.get("not_norm",False)
        self.torch_image = kwargs.get("torch_image",False)
        self.transform = kwargs.get("transform",True)

        self.input_mean = kwargs.get("input_mean",127.5)
        self.input_std = kwargs.get("input_std",127.5)

        if self.model_type=='onnx':
            #self.net = load_onnx.Onnx_cv(self.model_path,input_mean=127.5,input_std=127.5)
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=self.input_mean,input_std=self.input_std,torch_image = self.torch_image)
        elif self.model_type=='openvino':
            if self.load_multi:
                self.net = load_openvino.Openvino_multi(self.model_path,not_norm=self.not_norm,torch_image = self.torch_image,transform=self.transform)
            else:
                self.net = load_openvino.Openvino(self.model_path,not_norm=self.not_norm,torch_image = self.torch_image,device='CPU')

    def get(self,img,face,to_bgr=True,mask_flag=False):
        
        if mask_flag:
            if not 'aimg_mask' in face.keys():
                aimg_mask = face_align(img,LMARK_REF_ARC,face.land5,self.out_size)
                face.aimg_mask = aimg_mask
                aimg = aimg_mask
            else:
                aimg = face.aimg_mask

        else:
            if not 'aimg' in face.keys():
                aimg = face_align(img,LMARK_REF_ARC,face.land5,self.out_size)
                face.aimg = aimg
            else:
                aimg = face.aimg

        with torch.no_grad():
            feat = self.net(aimg)
            if isinstance(feat,list):
                feat = feat[0]
            feat = feat.flatten()
        feat = torch2numpy(feat)

        if mask_flag:
            face.feat_mask=feat
            return face.feat_mask

        else:
            face.feat = feat
            return face.feat

        

    def get_ref(self,img_path,to_bgr=True):
        img = read_image(img_path,to_bgr=to_bgr)
        
        with torch.no_grad():
            feat = self.net(img)
            if isinstance(feat,list):
                feat = feat[0]
            feat = feat.flatten()
        feat = torch2numpy(feat)
        
        return feat

    def get_features(self,img,need_trans,need_norm,need_torch):
        if len(img.shape)<4:
            img = np.expand_dims(img,0)
        if need_trans:
            img = np.transpose(img,(0,3,1,2))
        if need_norm:
            img = ((img / 255) - 0.5) / 0.5
        if need_torch:
            img = torch.from_numpy(img)
        with torch.no_grad():
            feat = self.net(img)
        feat = torch2numpy(feat)

        return feat

 
