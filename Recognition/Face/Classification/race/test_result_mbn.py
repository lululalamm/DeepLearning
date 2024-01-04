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

from backbones.get_models import get_model

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
    parser.add_argument("--net",type=str,default='mobilenetv3-small')
    parser.add_argument("--num_classes",default=6)
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--test_path",default="/data/shared/Face/FaceAGender/datasets/train/race/FairFace/aligned_fair_list_test_lb6.txt")
    parser.add_argument("--prefix",default="/data/shared/Face/FaceAGender/datasets/train/race/FairFace/aligned_fair/")
    parser.add_argument("--load_type",type=int,default=1)
    parser.add_argument("--save_name",default='mbnv3-small_l1')
    parser.add_argument("--result_base",default="./test_result/")
    parser.add_argument("--input_size",type=int,default=112)

    args = parser.parse_args()
    return args


args = get_args()

net=args.net
num_classes=args.num_classes
load_type = args.load_type
image_size = args.input_size

#label_dict={0:'White',1:'Black',2:'Southeast Asian',2:'East Asian',3:'Indian',4:'Middle Eastern',5:'Latino_Hispanic'}
label_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Middle Eastern',5:'Latino_Hispanic'}

test_path = args.test_path
prefix = args.prefix

save_name = args.save_name
save_base = args.result_base
if not os.path.exists(save_base):
    os.makedirs(save_base)

save_base = os.path.join(save_base,save_name)
if not os.path.exists(save_base):
    os.makedirs(save_base)
save_txt = os.path.join(save_base,"pred_list.txt")

# load model
model_path = args.model_path
backbone = get_model(net,num_classes=num_classes,load_type=load_type, \
                            input_size=image_size,pretrained=True)

load_weight = torch.load(model_path)
new_state_dict = OrderedDict()
for n, v in load_weight.items():
    name = n.replace("module.","") 
    # if load_type==3:
    #     name = n.replace("features.","") 
    new_state_dict[name] = v
backbone.load_state_dict(new_state_dict)

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
    