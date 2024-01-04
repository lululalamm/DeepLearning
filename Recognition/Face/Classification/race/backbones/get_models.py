import os
import numpy as np
import torch
from torchvision import models

from torch import nn

import sys
from .mobilenetv3 import get_mbnv3



def get_model(name,num_classes=6,load_type=1,input_size=112,pretrained=True):
    if name=='mobilenetv3-small':
        if load_type==1:
            model = models.mobilenet_v3_small(pretrained=pretrained)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            model = get_mbnv3(num_classes,'small',input_size=input_size,load_type=load_type)
    elif name=='mobilenetv3-large':
        if load_type==1:
            model = models.mobilenet_v3_large(pretrained=pretrained)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            model = get_mbnv3(num_classes,'large',input_size=input_size,load_type=load_type)

    return model