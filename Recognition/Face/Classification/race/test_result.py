import os
import numpy as np
import sys
import cv2
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module

from collections import OrderedDict

from backbones.linear import ArcfaceLinear_mbf

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_img = img.copy()
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img,ori_img

def get_args():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net",type=str,default='mbf')
    parser.add_argument("--num_features",default=512)
    parser.add_argument("--num_classes",default=6)
    parser.add_argument("--fp16",action='store_true',default=False)
    parser.add_argument("--prepath",default="")
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--test_path",default="/data/shared/Face/FaceAGender/datasets/train/race/FairFace/aligned_fair_list_test_lb6.txt")
    parser.add_argument("--prefix",default="/data/shared/Face/FaceAGender/datasets/train/race/FairFace/aligned_fair/")
    parser.add_argument("--save_name",default='sgd_221209')
    parser.add_argument("--result_base",default="/data/notebook/Face_Dataset_221206/results/221212_test/{}/")

    args = parser.parse_args()
    return args


args = get_args()

fp16=args.fp16
num_features=args.num_features
net=args.net
num_classes=args.num_classes
prepath=args.prepath

#label_dict={0:'White',1:'Black',2:'Southeast Asian',2:'East Asian',3:'Indian',4:'Middle Eastern',5:'Latino_Hispanic'}
label_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Middle Eastern',5:'Latino_Hispanic'}

test_path = args.test_path
prefix = args.prefix

save_name = args.save_name
save_base = args.result_base
save_base = save_base.format(save_name)
if not os.path.exists(save_base):
    os.makedirs(save_base)
save_txt = os.path.join(save_base,"pred_list.txt")

# load model
model_path = args.model_path
backbone = ArcfaceLinear_mbf(pretrained_path=prepath, net =net, num_class=num_classes, num_features=num_features,freeze=False,fp16=fp16)

load_weight = torch.load(model_path)
if type(load_weight)==OrderedDict:
    try:
        backbone.load_state_dict(load_weight)
    except:
        new_state_dict = OrderedDict()
        for n, v in load_weight.items():
            name = n.replace("module.","") 
            new_state_dict[name] = v
        backbone.load_state_dict(new_state_dict)
else:
    try:
        backbone.load_state_dict(load_weight.module.state_dict())
    except:
        backbone.load_state_dict(load_weight.state_dict())
backbone.eval()

# Start
lines = open(test_path,'r').readlines()

total = len(lines)
false_list=[]
true_ct=0

result_f = open(save_txt,'w')
result_f.writelines("model_path: {}\n".format(model_path))
result_f.writelines("test_path: {}\n".format(test_path))
result_f.writelines("Pred list\n")

for line in tqdm(lines):
    sp = line.strip().split(",")
    img_path,lb_name,lb = sp
    img_path = prefix+img_path
    lb = int(lb)
    
    img,ori_img = read_img(img_path)
    output = backbone(img)

    p,idx = torch.topk(output,1)
    p = np.array(p.detach())[0]
    idx = np.array(idx.detach())[0]
    pred_name = label_dict[idx[0]]

    new_line = "{},{},{},{}\n".format(img_path,lb,lb_name,pred_name)
    result_f.writelines(new_line)
    