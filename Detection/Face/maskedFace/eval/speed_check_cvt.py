import os 
import numpy as np
from PIL import Image
import torch
import cv2
from tqdm import tqdm

from totalface.model_zoo.model_common import load_tensorRT,load_openvino

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
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--model_type",type=str,default='trt',help='trt or vino')
    parser.add_argument("--img_size",type=int,default=112)
    parser.add_argument("--test_path",type=str,default="rmfd_mfr2/mfr2_2_scrfd_aligned/JoeBiden/JoeBiden_0001_mask.png")
    parser.add_argument("--num_iter",type=int,default=10000)
    args = parser.parse_args()
    return args


args = get_args()
model_path = args.model_path
model_type = args.model_type
img_size = args.img_size
test_path = args.test_path
num_iter = args.num_iter
print("model_path:",model_path)

test_img = image_preprocess(test_path,resize=img_size)

# load model
if model_type=='trt':
    model = load_tensorRT.TrtModel(model_path,torch_image=True,not_norm=True)
elif model_type=='vino':
    model_paths = [model_path+".xml",model_path+".bin"]
    model = load_openvino.Openvino(model_paths,not_norm=True,torch_image=True)


all_time=0
for i in tqdm(range(num_iter)):
    with torch.no_grad():
        s = time.time()
        output = model(test_img)[0]
        e = time.time()

    all_time+=((e-s)*1000)

speed = (all_time/num_iter)
print("Model: {}".format(model_type))
print("Model path: {}".format(model_path))
print("Time(batch): {} ms\n".format(speed))


