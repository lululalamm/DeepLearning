import os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict

from torchvision.models.efficientnet import efficientnet_b0


def get_model(name,pretrain='',freeze=True): # backbone
    if name=='b0':
        model = efficientnet_b0(pretrained=False)
    elif name=='v2_s':
        model = efficientnet_v2_s(pretrained=False)

    if pretrain:
        weight = torch.load(pretrain)
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            if "classifier.1" in k:
                continue
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict,strict=False)
    
    model = model.features
    
    if freeze:
        for name,param in model.named_parameters():
            param.requires_grad=False

    return model

class FAC(nn.Module):
    def __init__(self,net='b0',num_classes=8,pretrain_backbone='',freeze=True):
        super(FAC,self).__init__()

        self.base_model =  get_model(net,pretrain=pretrain_backbone,freeze=freeze)
        self.logits_layer = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                nn.Flatten(),
                                nn.Linear(1280,512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512,num_classes),
                                nn.Sigmoid()
                            )

    def forward(self,x):
        x = self.base_model(x)
        x = self.logits_layer(x)

        return x

def build_model(net='b0',num_classes=8,pretrain_backbone='',freeze=True):
    return FAC(net,num_classes,pretrain_backbone,freeze)


             

