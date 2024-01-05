import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from collections import OrderedDict
import argparse
from model import build_model
from get_config import get_config

from totalface.model_zoo.model_common import load_onnx,load_openvino

import time

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img


def preprocessing(aimg,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225],is_onnx=False):
    input_img = normalization(aimg,mean_list,std_list) # aimg is RGB
    if is_onnx:
        input_img = input_img.transpose(1,2,0)
        return input_img
    else:
        input_img = torch.tensor(np.expand_dims(input_img,0).astype(np.float32))
        return input_img

def get_args():
    parser = argparse.ArgumentParser(description='Speed check models')
    parser.add_argument('--cfg_path', type=str,default='configs/efficientNet_B0_celebA.py')
    parser.add_argument('--model_path',type=str,default='')
    parser.add_argument('--sample_path',type=str,default='sample.jpg')
    parser.add_argument('--iter_num',type=int,default=10000)


    args = parser.parse_args()

    return args

def main():
    args = get_args()

    cfg_path = args.cfg_path
    model_path = args.model_path
    sample_path = args.sample_path
    iter_num = args.iter_num

    # load model
    cfg = get_config(cfg_path)
    is_onnx=False

    if ".pth" in model_path: # torch
        model = build_model(cfg.network,cfg.num_classes,'',False)
        load_weight = torch.load(model_path)
        new_state_dict = OrderedDict()
        for n, v in load_weight.items():
            name = n.replace("module.","") 
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        _ = model.eval()
        model_format = "torch"

    elif ".onnx" in model_path: # onnx
        model = load_onnx.Onnx_session(model_path,input_mean=0.0, input_std=1.0,output_sort=True,onnx_device='cuda')
        is_onnx=True
        model_format="onnx"

    elif ".vino" in model_path:
        model_path = [model_path.split(".vino")[0]+".xml",model_path.split(".vino")[0]+".bin"]
        model = load_openvino.Openvino(model_path,not_norm=True,torch_image=True,device='CPU')
        model_format = "openvino"

    img = cv2.imread(sample_path)
    input_img = preprocessing(img,is_onnx=is_onnx)
    total = 0

    for i,num in enumerate(tqdm(range(iter_num))):
        st = time.time()
        out = model(input_img)
        ed = time.time()

        if i==0:
            continue
        total+=((ed-st)*1000)


    print("model_path:",model_path)
    print("model_format:",model_format)
    print("mean time:",(total/(iter_num-1)),"ms")

if __name__ == "__main__":
    main()

