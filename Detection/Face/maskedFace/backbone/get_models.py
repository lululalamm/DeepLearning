import os
import numpy as np
import torch
from torchvision import models

from torch import nn

import sys
from mobilefacenet import get_mbf
from mobilenetv3 import get_mbfv3
from mnasnet import get_mnas


def get_model(name,num_classes=2,load_type=1,input_size=112,mnas_size='075'):
    if name=='mnasnet':
        # model load
        if load_type==1:
            if mnas_size=='075':
                model = models.mnasnet0_75(pretrained=False)
            elif mnas_size=='10':
                model = models.mnasnet1_0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs,num_classes)
        else:
            model = get_mnas(num_classes,mnas_size)
    elif name=='mobilenetv3-small':
        if load_type==1:
            model = models.mobilenet_v3_small(pretrained=False)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            model = get_mbfv3(num_classes,'small',input_size=input_size,load_type=load_type)
    elif name=='mobilenetv3-large':
        if load_type==1:
            model = models.mobilenet_v3_large(pretrained=False)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            model = get_mbfv3(num_classes,'large',input_size=input_size,load_type=load_type)
    elif name=='mobilefacenet':
        model = get_mbf(num_classes=num_classes,fp16=False,num_features=512,input_size=input_size)

    return model