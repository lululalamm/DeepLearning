import argparse

import os
import numpy as np
from PIL import Image
import cv2
import torch
import random
import pandas as pd
from tqdm import tqdm
from totalface_cpu.model_zoo.model_common import load_onnx

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

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def read_image(path,mean,std,resize=128,prefix='',ori_return=False):
    img = cv2.imread(prefix+path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cropped = img

    input_img = cv2.resize(cropped,(resize,resize),interpolation=cv2.INTER_CUBIC)
    input_img = normalization(input_img,mean,std)
    input_img = np.transpose(input_img,(1,2,0))
    
    if ori_return:
        return input_img, img
    else:
        return input_img



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_format",default="safas/csv/{}_crop_list_test.csv")
    parser.add_argument("--model_path",default="")
    parser.add_argument('--db', default='wild',help='wild or celebA or casia')
    parser.add_argument('--resize',type=int,default=128)
    parser.add_argument('--prefix',type=str,default="/data/")

    args = parser.parse_args()
    return args



args = get_args()

csv_format = args.csv_format
db = args.db
model_path = args.model_path
resize = args.resize
prefix = args.prefix

csv_path = csv_format.format(db)

# model load 
model = load_onnx.Onnx_session(model_path,input_mean=0.0, input_std=1.0,output_sort=True)
# input (1,3,128,128)
# output (1,2)

# result label
pred_dict={0:'real',1:'fake'}

# image param
mean=[0.5931, 0.4690, 0.4229]
std=[0.2471, 0.2214, 0.2157]

# dataset load
df = pd.read_csv(csv_path,header=None)

paths = df[0]
ids = df[1]
labels_str = df[2]

save_base = os.path.join(*model_path.split("/")[:-1])
save_path = os.path.join(save_base,"test_{}.txt".format(db))


# start
gt=[]
pred=[]
real_scores=[]

with open(save_path,'w') as f:
    f.writelines("path,gt_str,lb_gt,real_score,fake_score,pred_idx\n")
    for i in tqdm(range(len(paths))):
        path = paths[i]
        gt_str = labels_str[i]

        input_img = read_image(path,mean,std,resize=resize,prefix=prefix)

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
