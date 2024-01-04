import torch
import math
from torch import nn
import torch.nn.functional as F

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

class FocalLoss(nn.Module):
    def __init__(self,classes_num,gamma=2.,alpha=.25,e=0.1):
        super(FocalLoss,self).__init__()
        
        self.gamma = gamma
        self.classes_num = classes_num
        self.alpha = alpha
        self.e = e
    
    def forward(self,prediction,target):
        zeros = torch.zeros_like(prediction,dtype=torch.double)
        one_minus_p = torch.where(torch.greater(target,zeros),target - prediction, zeros)
        FT = -1 * (one_minus_p ** self.gamma) * torch.log(torch.clip(prediction,1e-6,1.0))
        
        classes_weight = torch.zeros_like(prediction, dtype=torch.double)
        
        total_num = float(sum(self.classes_num))
        classes_w_t1 = [ (total_num / ff if ff!=0 else 0.0) for ff in self.classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = torch.tensor([ ff/sum_ for ff in classes_w_t1 ])
        classes_w_tensor = classes_w_t2.type(torch.double)
        classes_w_tensor = classes_w_tensor.to('cuda')
        classes_weight += classes_w_tensor
        
        alpha = torch.where(torch.greater(target,zeros),classes_weight,zeros)
        
        balanced_fl = self.alpha * FT
        balanced_fl = torch.mean(balanced_fl)
        
        nb_classes = len(self.classes_num)
        final_loss = (1-self.e) * balanced_fl + self.e * F.binary_cross_entropy_with_logits(torch.ones_like(prediction)/nb_classes, prediction)
        return final_loss