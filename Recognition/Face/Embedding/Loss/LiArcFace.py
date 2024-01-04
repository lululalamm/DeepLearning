import math
import numpy as np
from math import pi

import torch
from torch import nn
import torch.nn.functional as F


class LiArcFace(nn.Module):
    def __init__(self, num_classes, emb_size=512, m=0.45, s=64.0):
        super().__init__()

        self.m = m
        self.s = s

    def forward(self, input, label,weight): # input, weight not normalized
        self.weight = weight
        W = F.normalize(self.weight)
        input = F.normalize(input)
        cosine = input @ W.t()
        theta = torch.acos(cosine)
        m = torch.zeros_like(theta)
        m.scatter_(1, label.view(-1, 1), self.m)
        logits = self.s * (pi - 2 * (theta + m)) / pi
        return logits
