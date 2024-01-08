# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

from totalface_cpu.model_zoo.get_models import get_detection_model
from totalface_cpu.face.get_result import get_detection

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet': MultiFTNet
}

def get_paths(path):
    model_paths=[ path+".xml",path+".bin" ]
    return model_paths


class Detection:
    def __init__(self):
        
        self.dt_name='scrfd'
        self.dt_path = get_paths("scrfd_10g_bnkps_quantINT8")
        self.input_size=(640,640)
        self.detector = get_detection_model(self.dt_name,self.dt_path,load_multi=False)
        self.detector_confidence = 0.5

    def get_bbox(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces = get_detection(self.dt_name,self.detector,img,thresh=self.detector_confidence,input_size=self.input_size)
        if len(faces)<1:
            return []
        bbox_ori = faces[0]['bbox']
        bbox = [int(bbox_ori[0]), int(bbox_ori[1]), int(bbox_ori[2]-bbox_ori[0]+1), int(bbox_ori[3]-bbox_ori[1]+1)]
        return bbox # x,y,w,h


class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path,num_classes=3,h=80,w=80,model_type=''):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type = h,w,model_type
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size,num_classes=num_classes).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path,num_classes=3,h=80,w=80,model_type='',norm_input=False):
        if norm_input:
            test_transform = trans.Compose([
                trans.ToTensor(),
            ])
        else:
            test_transform = trans.Compose([
                trans.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path,num_classes=num_classes,h=h,w=w,model_type=model_type)
        self.model.eval()
        with torch.no_grad():
            result,result_ft = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result,result_ft











