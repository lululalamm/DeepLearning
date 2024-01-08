# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:14
# @Author : zhuying
# @Company : Minivision
# @File : MultiFTNet.py
# @Software : PyCharm
from torch import nn
import torch
import torch.nn.functional as F
from src.model_lib.MiniFASNet import MiniFASNetV1,MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.model_lib.mobilefacenet import get_mbf
from src.model_lib.iresnet import iresnet18, iresnet34, iresnet50,iresnet100,iresnet200


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128, eps=1e-4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64, eps=1e-4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, img_channel=3, num_classes=2, embedding_size=128, conv6_kernel=(5, 5)):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.model = MiniFASNetV2SE(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        # if self.training:
        #     ft = self.FTGenerator(x)
        #     return cls, ft
        # else:
        #     return cls
        ft = self.FTGenerator(x)
        return cls, ft


class MultiFTNet_mbf(nn.Module):
    def __init__(self, num_classes=3, embedding_size=128):
        super(MultiFTNet_mbf, self).__init__()
        self.num_classes = num_classes
        self.model = get_mbf(fp16=False,num_features=embedding_size,num_classes=num_classes)
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls

resnet_mapping={
    'r18':iresnet18,
    'r34':iresnet34,
    'r50':iresnet50,
    'r100':iresnet100,
    'r200':iresnet200
}

from torch.nn import Module,Linear, Conv2d, BatchNorm1d, BatchNorm2d

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Linear_block(Module):
    def __init__(self, in_c=512, out_c=512, kernel=(5, 5), stride=(1, 1), padding=(0, 0), groups=512):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                           groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c, eps=1e-4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiFTNet_resnet(nn.Module):
    def __init__(self, num_classes=3, embedding_size=128, network='r100',dropout=0,fp16=False):
        super(MultiFTNet_resnet, self).__init__()
        self.num_classes = num_classes
        model = resnet_mapping[network](num_features=embedding_size,dropout=dropout,fp16=fp16)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.prelu = model.prelu
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()
        self.embedding_size = embedding_size

        # add layer
        #self.conv_6_dw = Linear_block()
        self.flatten = Flatten()
        self.linear = nn.Linear(12800, embedding_size)
        #self.linear = Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size, eps=1e-4)
        self.drop = nn.Dropout(p=dropout)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x) # 128
        x1 = self.layer3(x)
        x1 = self.layer4(x1)
        #x1 = self.conv_6_dw(x1) # add 
        x1 = self.flatten(x1)  # resnet
        x1 = self.linear(x1)
        x1 = self.bn(x1)
        x1 = self.drop(x1)
        cls = self.prob(x1)

        if self.training:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls
