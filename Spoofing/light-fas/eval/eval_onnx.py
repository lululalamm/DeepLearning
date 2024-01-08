import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import h5py
import random

from totalface_cpu.model_zoo.model_common import load_onnx

from tqdm import tqdm
from sklearn.metrics import auc, roc_curve


def read_image(path,bbox,mean,std,resize=128,prefix=''):
    img = cv2.imread(prefix+path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cropped = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]

    input_img = cv2.resize(cropped,(resize,resize),interpolation=cv2.INTER_CUBIC)
    input_img = normalization(input_img,mean,std)
    input_img = np.transpose(input_img,(1,2,0))

    return input_img


def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

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

    acer = (apcer+bpcer)/2
    acc = float(tp + tn) / len(pred)
    
    #BCR = (tp / (tp + fn) + tn / (tn + fp))/2
    #hter = 1-BCR

    return tn,tp,fn,fp,tpr,fpr,apcer,bpcer,acer,acc#,hter

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




# model 
model_path = "./pretrained/anti-spoof-mn3.onnx"
model = load_onnx.Onnx_session(model_path,input_mean=0.0, input_std=1.0,output_sort=True)
# input (1,3,128,128)
# output (1,2)

mean=[0.5931, 0.4690, 0.4229]
std=[0.2471, 0.2214, 0.2157]
resize=128

# dataset
h5_path = "FaceInTheWild/results_scrfd_10g/wild_test.h5"
hf = h5py.File(h5_path,'r')

paths = hf['path']
bboxs = hf['bboxs']
labels = hf['labels']

prefix = ""
save_path = "./test_wild.txt"

pred_dict={0:'real',1:'fake'}


# start
gt=[]
pred=[]
real_scores=[]

with open(save_path,'w') as f:
    f.writelines("path,gt_str,lb_gt,real_score,fake_score,pred_idx\n")
    for i in tqdm(range(len(paths))):
        path = paths[i]
        bbox = bboxs[i]
        gt_str = labels[i]

        input_img = read_image(path,bbox,mean,std,resize=resize,prefix=prefix)

        # run
        out = model(input_img)[0][0] # prob (real,fake)
        pred_idx = np.argmax(out)
        if pred_idx==0: #real
            pred.append(1)
        else:
            pred.append(0)

        if gt_str=='real':
            lb_gt=1
        else:
            lb_gt=0
        gt.append(lb_gt)

        real_scores.append(out[0])

        new_line = "{},{},{},{},{},{}\n".format(path,gt_str,lb_gt,out[0],out[1],pred_idx)
        f.writelines(new_line)


# eval
tn,tp,fn,fp,tpr,fpr,apcer,bpcer,acer,acc = eval(gt,pred)
frr,far,eer,hter = eval_eer(gt,real_scores)

print("Dataset: wild")
print("tn: {} / tp: {} / fn: {} / fp: {}".format(tn,tp,fn,fp))
print("tpr: {} / fpr: {}".format(tpr,fpr))
#print("frr: {} / far: {}".format(frr,far))
print("apcer: {} / bpcer: {} / acer: {}".format(apcer,bpcer,acer))
print("eer: {} / hter: {}".format(eer,hter))
print("acc: {}".format(acc))
