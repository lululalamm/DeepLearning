# torch                        1.12.1+cu113
# torchaudio                   0.12.1+cu113
# torchvision                  0.13.1+cu113

import os
import torch
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.models.efficientnet import efficientnet_v2_s
from collections import OrderedDict



def get_efficientNet(name,num_features=512,pretrain=''):
    if name=='b0':
        model = efficientnet_b0(pretrained=False,num_classes=num_features)
        # weight = torch.load("./efficientnet_b0_rwightman-3dd342df.pth")
    elif name=='v2_s':
        model = efficientnet_v2_s(pretrained=False,num_classes=num_features)
        # weight = torch.load("./efficientnet_v2_s-dd5fe13b.pth")


    if pretrain:
        weight = torch.load(pretrain)
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            if "classifier.1" in k:
                continue
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict,strict=False)

    return model