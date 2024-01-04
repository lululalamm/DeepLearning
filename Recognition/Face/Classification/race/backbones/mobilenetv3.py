import os
import numpy as np
import torch
from torchvision import models

from torch import nn

class MobileNetV3_load2(torch.nn.Module):
    def __init__(self,num_classes=2,size='small',sub_feature=128):
        super(MobileNetV3_load2, self).__init__()
        print("mbnetv3 load2:",size)

        if size=='small':
            model = models.mobilenet_v3_small(pretrained=True)
        else:
            model = models.mobilenet_v3_large(pretrained=True)

        
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, sub_feature)

        self.base_net = model

        self.logits_layer = nn.Sequential(
            #nn.Linear(num_ftrs,128), 
            nn.LeakyReLU(), 
            nn.Linear(sub_feature , num_classes)
        )
    def forward(self, x):
        x = self.base_net(x)
        x = self.logits_layer(x)
        return x

class MobileNetV3_load3(torch.nn.Module):
    def __init__(self,num_classes=2,size='small',input_size=112):
        super(MobileNetV3_load3, self).__init__()
        print("mbnetv3 load3:",size)

        if size=='small':
            model = models.mobilenet_v3_small(pretrained=True)
        else:
            model = models.mobilenet_v3_large(pretrained=True)

        self.base_net = model.features # not classifier

        if input_size==112:
            pool_kernel = 4
        elif input_size==160:
            pool_kernel = 5
        elif input_size==224:
            pool_kernel = 7
        
        self.logits_layer = nn.Sequential(
                    nn.AvgPool2d([pool_kernel,pool_kernel]),
                    nn.Flatten(),
                    nn.Linear(960,128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128,num_classes)
                )

    def forward(self, x):
        x = self.base_net(x)
        x = self.logits_layer(x)
        return x



def get_mbnv3(num_classes,size,input_size=112,load_type=2):
    if load_type==2:
        return MobileNetV3_load2(num_classes,size)
    elif load_type==3:
        return MobileNetV3_load3(num_classes,size,input_size)