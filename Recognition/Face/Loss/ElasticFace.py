import math
import numpy as np

import torch
from torch import nn

class ElasticArcFace(nn.Module):
    def __init__(self,s=64.0, m=0.50,std=0.05,plus=False):
        super(ElasticArcFace, self).__init__()
        self.s = s
        self.m = m
        self.std=std
        self.plus=plus

    def forward(self, logits, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device) # dim 2
        margin = torch.normal(mean=self.m, std=self.std, size=label[index].size(), device=logits.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = logits[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index], margin)
        logits.acos_()
        logits[index] += m_hot
        logits.cos_().mul_(self.s)
        return logits



class ElasticCosFace(nn.Module):
    def __init__(self,s=64.0, m=0.35,std=0.0125, plus=False):
        super(ElasticCosFace, self).__init__()
        self.s = s
        self.m = m
        self.std=std
        self.plus=plus


    def forward(self, logits, label):
        
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=logits.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = logits[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        logits[index] -= m_hot
        ret = logits * self.s
        return 