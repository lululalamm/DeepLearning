from collections import OrderedDict
from torchvision import transforms as T
import torch.nn as nn
import torch

import cv2
import numpy as np

from ...data.image import read_torchImage, read_image
from ...utils.util_warp import face_align
from ...data.constant import LMARK_REF_ARC

from ..backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from ..backbones.mobilefacenet import get_mbf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_arcface(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)
    else:
        raise ValueError()

class Arcface_torch:
    def __init__(self,model_path,out_size=112,num_features=512,network='r50',fp16=False):

        self.out_size=out_size
        self.num_features = num_features
        self.fp16=fp16
        self.model_path = model_path
        self.network = network

        self.net = get_arcface(self.network,num_features=self.num_features,fp16=self.fp16)
    
        load_weight = torch.load(self.model_path)

        if type(load_weight)==OrderedDict:
            try:
                self.net.load_state_dict(load_weight)
            except:
                new_state_dict = OrderedDict()
                for n, v in load_weight.items():
                    name = n.replace("module.","") 
                    new_state_dict[name] = v
                self.net.load_state_dict(new_state_dict)
        else:
            try:
                self.net.load_state_dict(load_weight.module.state_dict())
            except:
                self.net.load_state_dict(load_weight.state_dict())

        self.net.to(device)
        self.net.eval()  

    def get(self,img,face,to_bgr=True):

        if not 'aimg' in face.keys():
            aimg = face_align(img,LMARK_REF_ARC,face.land5,self.out_size)
            face.aimg = aimg
        else:
            aimg = face.aimg

        torch_img = read_torchImage(aimg,to_bgr)
        torch_img = torch_img.to(device)
        with torch.no_grad():
            feat = self.net(torch_img)
            feat = feat.flatten()
        feat = feat.cpu().numpy()
        face.feat = feat

        return face.feat

    def get_ref(self,img_path,to_bgr=True):
        torch_img = read_torchImage(img_path,to_bgr)
        torch_img = torch_img.to(device)
        with torch.no_grad():
            feat = self.net(torch_img)
            feat = feat.flatten()
        feat = feat.cpu().numpy()
        
        return feat



class Arcface_onnx:
    def __init__(self,model_path,out_size=112):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.layer_names = self.net.getLayerNames()
        self.output_names = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.out_size = out_size

    def get(self,img,face,to_bgr=False):
        if not 'aimg' in face.keys():
            aimg = face_align(img,LMARK_REF_ARC,face.land5,self.out_size)
            face.aimg = aimg
        else:
            aimg = face.aimg
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(aimg, 1.0/127.5, input_size, (127.5, 127.5, 127.5), swapRB=to_bgr)

        self.net.setInput(blob)

        feat = self.net.forward(self.output_names)[0]
        feat = feat.flatten()
        face.feat = feat

        return face.feat

    def get_ref(self,img_path,to_bgr=True):
        img = read_image(img_path)
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/127.5, input_size, (127.5, 127.5, 127.5), swapRB=to_bgr)
        self.net.setInput(blob)

        feat = self.net.forward(self.output_names)[0]
        feat = feat.flatten()

        return feat



        





