import os 
import numpy as np
from PIL import Image
import torch
import cv2
from tqdm import tqdm


from backbone.get_models import get_model

from collections import OrderedDict

import random
import argparse
import time

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def image_preprocess(img_path,resize=224,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize))
    
    img = normalization(img,mean_list,std_list)
    
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default='mobilenetv3-small',help="mnasnet075 / mobilenetv3-small / mobilefacenet")
    parser.add_argument("--model_path",type=str,default="./save_models/mobilenetv3-small_facemask_230227/last.pth")
    parser.add_argument("--num_classes",type=int,default=2)
    parser.add_argument("--test_path",type=str,default="rmfd_mfr2/mfr2_2_scrfd_aligned/JoeBiden/JoeBiden_0001_mask.png")
    parser.add_argument("--num_iter",type=int,default=10000)
    parser.add_argument("--load_type",type=int,default=1)
    args = parser.parse_args()
    return args


args = get_args()
model_name = args.model_name
model_path = args.model_path
num_classes = args.num_classes
test_path = args.test_path
num_iter = args.num_iter
load_type = args.load_type

if "mobilefacenet"==model_name:
    test_img = image_preprocess(test_path,resize=112)
else:
    test_img = image_preprocess(test_path)



# load model
model = get_model(model_name,num_classes,load_type=load_type)

load_weight = torch.load(model_path)
new_state_dict = OrderedDict()
for n, v in load_weight.items():
    name = n.replace("module.","") 
    new_state_dict[name] = v

if model_name=='mnasnet075':
    if not hasattr(new_state_dict, '_metadata'):
         setattr(new_state_dict, '_metadata', OrderedDict({'version': 2}))


model.load_state_dict(new_state_dict)
_ = model.eval()

all_time=0
for i in tqdm(range(num_iter)):
    with torch.no_grad():
        s = time.time()
        output = model(test_img).numpy()
        e = time.time()

    all_time+=((e-s)*1000)

speed = (all_time/num_iter)
print("Model: {}".format(model_name))
print("Model path: {}".format(model_path))
print("Time(batch): {} ms\n".format(speed))


