import math
import numpy as np
from math import pi

import torch
from torch import nn
import torch.nn.functional as F

class MarginDistillation(torch.nn.Module):
    def __init__(self, s=64.0):
        super(MarginDistillation, self).__init__()
        self.scale = s
        self.easy_margin = False


    def forward(self,margin:torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        target_margin = margin[index]

        cos_m = torch.cos(target_margin)
        sin_m = torch.sin(target_margin)
        theta = torch.cos(math.pi - target_margin)
        sinmm = torch.sin(math.pi - target_margin) * target_margin

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))

        cos_theta_m = target_logit * cos_m - sin_theta * sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > theta, cos_theta_m, target_logit - sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits