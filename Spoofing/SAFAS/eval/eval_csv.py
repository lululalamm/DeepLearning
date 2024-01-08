import argparse

import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from networks import get_model
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

def eval_eer(gt,scores,get_hter=False):
    fpr, tpr, thresholds = roc_curve(gt, scores, pos_label=1)
    frr = 1-tpr
    far = fpr
    
    min_index = np.nanargmin(np.absolute((frr - far)))
    far_eer = far[min_index]
    frr_eer = frr[min_index]
    eer = (frr_eer+far_eer)/2
    
    hter_list=[]
    for i in range(len(frr)):
        hter_list.append((frr[i]+far[i])/2)

    hter = np.mean(hter_list)
    
    if get_hter:
        return frr,far,eer,hter,hter_list
    else:
        return frr,far,eer,hter

def get_err_threhold(gt,scores):
    fpr, tpr, thresholds = roc_curve(gt, scores, pos_label=1)

    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = thresholds[right_index]
    err = fpr[right_index]    
    return err, best_th, right_index

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def str2bool(x):
    return x.lower() in ('true')

def get_random_crop(height, width, scale, ratio):
    area = height * width
    log_ratio = np.log(ratio)
    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = int(np.random.uniform(0, height - h + 1))
            j = int(np.random.uniform(0, width - w + 1))
            return i, j, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_format",default="safas/csv/{}_{}_list_test.csv")
    parser.add_argument("--ckpt_path",default="./results/~")
    parser.add_argument('--db', default='wild',help='wild or celebA or casia')
    parser.add_argument('--model_type',default='ResNet18_lgt')
    parser.add_argument('--cls_num',type=int,default=2)
    parser.add_argument('--normfc',default=False,type=str2bool)
    parser.add_argument('--usebias',default=True,type=str2bool)
    parser.add_argument('--feat_loss',default='supcon')
    parser.add_argument('--save_name',default='',type=str)
    parser.add_argument('--method',default='crop',type=str,help='crop or align or crop_resize')
    parser.add_argument('--img_size',default=256,type=int)


    args = parser.parse_args()
    return args


args = get_args()

# test data
db = args.db
method = args.method
csv_format = args.csv_format
csv_path = csv_format.format(db,method.replace("_resize",""))
img_size = args.img_size

print("load csv:",csv_path)

# model
model_type = args.model_type
cls_num = args.cls_num
normfc = args.normfc
usebias = args.usebias
feat_loss = args.feat_loss
save_name = args.save_name

ckpt_path = args.ckpt_path
if save_name:
    save_name = save_name+"_{}.txt"
else:
    save_name = "result_{}.txt"

save_path = os.path.join(*ckpt_path.split("/")[:-1],save_name.format(db))

print("model type: {} / cls num: {} / normfc: {} / usebias: {} / feat loss: {} ".format(model_type,cls_num,normfc,usebias,feat_loss))
print("load:",ckpt_path)


# get model
max_iter=-1
total_cls_num=2
model = get_model(model_type, \
                  max_iter, total_cls_num, pretrained=False, \
                  normed_fc=normfc, use_bias=usebias, \
                  simsiam=True if feat_loss == 'simsiam' else False)

ckpt = torch.load(ckpt_path)
state_dict = ckpt['state_dict']

new_state_dict = OrderedDict()
for n, v in state_dict.items():
    name = n.replace("module.","") # dataparallel
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)


_ = model.eval()

# Start
df = pd.read_csv(csv_path,header=None)

# for eval
gt=[]
pred=[]
real_scores=[]
ct=[0,0]

for val in tqdm(df.values):
    img_path, id_val, lb_str = val

    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    if method=='crop_resize':
        img = cv2.resize(img,(img_size,img_size))
    elif method=='crop': # randomresizedcrop
        ori_h,ori_w,_ = img.shape
        i, j, h, w = get_random_crop(ori_h,ori_w,scale=(0.9,0.9),ratio=(1.,1.))
        img = img[i:i+h,j:j+w]
        img = cv2.resize(img,(256,256))

    img = normalization(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img,axis=0)
    img = torch.tensor(img)


    # run
    _, penul_feat, logit = model(img)
    score = logit.squeeze().item() # real score

    if score>=0.5:
        pred_lb = 'real'
        pred.append(1)
    else:
        pred_lb = 'fake'
        pred.append(0)

    real_scores.append(score)

    if lb_str=='real':
        gt.append(1)
    else:
        gt.append(0)


tn,tp,fn,fp,tpr,fpr,apcer,bpcer,acer,acc = eval(gt,pred)
frr,far,eer,hter,hter_list = eval_eer(gt,real_scores,get_hter=True)
best_eer, best_th, right_index = get_err_threhold(gt,real_scores)
best_th_hter = hter_list[right_index]
    
print("Dataset:",db)
print("tn: {} / tp: {} / fn: {} / fp: {}".format(tn,tp,fn,fp))
print("tpr: {} / fpr: {}".format(tpr,fpr))
#print("frr: {} / far: {}".format(frr,far))
print("apcer: {} / bpcer: {} / acer: {}".format(apcer,bpcer,acer))
print("eer: {} / hter: {}".format(eer,hter))
print("acc: {}".format(acc))
print("best threshold(real): {} / best_threshold eer: {} / best threshold hter: {}".format(best_th,best_eer,best_th_hter))

with open(save_path,'w') as f:
    f.writelines("Dataset: {}\n".format(db))
    f.writelines("tn: {} / tp: {} / fn: {} / fp: {}\n".format(tn,tp,fn,fp))
    f.writelines("tpr: {} / fpr: {}\n".format(tpr,fpr))
    f.writelines("apcer: {} / bpcer: {} / acer: {}\n".format(apcer,bpcer,acer))
    f.writelines("eer: {} / hter: {}\n".format(eer,hter))
    f.writelines("acc: {}\n".format(acc))
    f.writelines("best threshold(real): {} / best_threshold eer: {} / best threshold hter: {}\n".format(best_th,best_eer,best_th_hter))





