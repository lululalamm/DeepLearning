import os
import numpy as np
import torch
from torchvision import models

from torch import nn

class MNasNet(torch.nn.Module):
    def __init__(self,num_classes=2,size='075',sub_feature=128):
        super(MNasNet, self).__init__()

        if size=='075':
            model = models.mnasnet0_75(pretrained=False)
        elif size=='10':
            model = models.mnasnet1_0(pretrained=False)
        else:
            print("size error....")
            return None

        num_ftrs = model.classifier[1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, sub_feature)
        
        self.base_net = model
        
        print(num_ftrs)
        self.logits_layer = nn.Sequential(
            #nn.Linear(num_ftrs,128), 
            nn.LeakyReLU(), 
            nn.Linear(128 , num_classes)
        )

    def forward(self, x):
        x = self.base_net(x)
        x = self.logits_layer(x)
        return x


def get_mnas(num_classes,size):
    return MNasNet(num_classes,size)


