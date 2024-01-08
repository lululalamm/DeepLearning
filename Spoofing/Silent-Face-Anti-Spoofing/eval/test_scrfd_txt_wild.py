# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm

from src.anti_spoof_predict_scrfd import AntiSpoofPredict
from src.generate_patches import CropImage
import random

warnings.filterwarnings('ignore')

import time
from datetime import datetime
import json


from sklearn.metrics import auc, roc_curve

def eval(gt,pred):
    gt = np.array(gt)
    pred = np.array(pred)
    # if ... true 1 (real) / false 0 (spoof)
    tp = np.sum(np.logical_and(gt,pred))
    tn = np.sum(np.logical_and(np.logical_not(gt),np.logical_not(pred)))
    fp = np.sum(np.logical_and(np.logical_not(gt),pred))
    fn = np.sum(np.logical_and(gt,np.logical_not(pred)))

    tpr = float(tp) / float(tp + fn)
    fpr = float(fp) / float(fp + tn)
    apcer = float(fp)/float(fp+tn) # apcer fpr
    bpcer = float(fn)/float(fn+tp) # bpcer fnr
    acc = float(tp + tn) / len(pred)

    acer = (apcer+bpcer)/2
    
    #BCR = (tp / (tp + fn) + tn / (tn + fp))/2
    #hter = 1-BCR

    return tn,tp,fn,fp,tpr,fpr,apcer,bpcer,acer,acc#,hter

from sklearn.metrics import auc, roc_curve

def eval_eer(gt,scores):
    fpr, tpr, thresholds = roc_curve(gt, scores, pos_label=1)
    frr = 1-tpr
    far = fpr
    
    min_index = np.nanargmin(np.absolute((frr - far)))
    far_eer = far[min_index]
    frr_eer = frr[min_index]
    eer = (frr_eer+far_eer)/2
    
    hter=[]
    for i in range(len(frr)):
        hter.append((frr[i]+far[i])/2)

    hter = np.mean(hter)
    
    return frr,far,eer,hter


def test(args):

    image_txt, model_path, device_id = args.image_txt, args.model_path, args.device_id
    model_type ,scale,num_classes,norm_input = args.model_type, args.scale, args.num_classes, args.norm_input

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    if scale=='org':
        scale = None
        w_input=80
        h_input=60

    else:
        scale = float(scale)
        w_input=80
        h_input=80

    lines = open(image_txt,'r').readlines()

    label_list = ['real','print','replay','3d']
   

    box_times=0
    times=0
    pretimes=0

    gt=[]
    pred=[]
    save_scores=[]
    real_scores=[]
    score_save = os.path.join(*model_path.split("/")[:-1])+"/scores_{}.npy"
    
    for line in tqdm(lines):
        sp = line.strip().split(",")
        ori_path,org_path,_,_,_,_,_,mask,_,label_str = sp
        ori_path = "/data/notebook/"+ori_path
        label_str = label_list.index(label_str)
    
        #if total_each[label_gt]>=5000: continue
        #total_each[label_gt]+=1

        box_s = time.time()
        image = cv2.imread(ori_path)
        image_bbox = model_test.get_bbox(image)
        if len(image_bbox)<1:
            continue
        box_e = time.time()
        box_times+=((box_e-box_s)*1000)
            
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }

        pre_st = time.time()
        img = image_cropper.crop(**param)
        pre_end = time.time()
        pretimes += ((pre_end-pre_st)*1000)

        st = time.time()
        prediction,ft_map = model_test.predict(img, model_path,num_classes = num_classes,h=h_input,w=w_input,model_type=model_type,norm_input=args.norm_input)
        ed = time.time()
        times +=((ed-st)*1000)

        label_pred = np.argmax(prediction)
        if not label_str==0: # fake
            label_gt==0 # 0 or 1
        else:
            label_gt=1 # real True(1), spoof False(0)

        save_scores.append([prediction[0],int(label_str)]) # real 0 / spoof 1
        real_scores.append([prediction[0][0]])

        gt.append(label_gt)

        if label_pred==0: # real
            pred.append(1) # True
        else:
            pred.append(0)
    
    total = len(pred)
    print(">> detection times: {} ms".format((box_times/total)))
    print(">> preprocessing times: {} ms".format((pretimes/total)))
    print(">> infer times: {} ms".format((times/total)))

    print("wild test results ... ")
    tn,tp,fn,fp,tpr,fpr,apcer,bpcer,acer,acc = eval(gt,pred)
    frr,far,eer,hter = eval_eer(gt,real_scores)
    print(">> ACC:",acc)
    print(">> APCER:",apcer)
    print(">> BPCER:",bpcer)
    print(">> ACER:",acer)
    print(">> HTER:",hter)
    print(">> eer:",eer)

    np.save(score_save.format("all"),np.array(save_scores))



if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_logs/snapshot/Anti_Spoofing_2.7_80x80/2023-02-11-14-44_Anti_Spoofing_2.7_80x80_model_iter-31150.pth",
        help="model path")
    parser.add_argument(
        "--image_txt",
        type=str,
        default="FaceInTheWild/unmask/Validation/image_list_org_1_80x60_new_unmask.txt",
        help="image txt path used to test")
    parser.add_argument("--scale",type=str,default='2.7',help=['org','1','2.7','4'])
    parser.add_argument("--model_type",type=str,default='MultiFTNet',
                                    help=['MultiFTNet','MultiFTNet_resnet'])
    parser.add_argument("--resnet_net",type=str,default='r100',help='if use resnet ..')
    parser.add_argument('--num_classes',type=int,default=4)
    parser.add_argument('--samples',type=int,default=0)
    parser.add_argument('--norm_input',action='store_true',default=False)

    args = parser.parse_args()
    test(args)
