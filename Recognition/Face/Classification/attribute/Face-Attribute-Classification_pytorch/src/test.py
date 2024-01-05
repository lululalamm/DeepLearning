import os
import numpy as np
import cv2
import torch
import random
import h5py
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from torcheval.metrics import BinaryAccuracy,AUC

from evaluate import Precision,Recall,F1_score
from model import build_model
from get_config import get_config

# if need detection N align
from totalface_cpu.model_zoo.get_models import get_detection_model
from totalface_cpu.face.get_result import get_detection

from totalface_cpu.utils.util_warp import face_align
from totalface_cpu.data.constant import LMARK_REF_ARC

import time


def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def align(img,land,ref_land,input_size=224):
    aimg = face_align(img,ref_land,land,input_size)
    return aimg

def preprocessing(aimg,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    input_img = normalization(aimg,mean_list,std_list) # aimg is RGB
    input_img = torch.tensor(np.expand_dims(input_img,0).astype(np.float32))
    
    return input_img

def eval(outs,gts):
    
    precision_m = Precision()
    recall_m = Recall()
    bAcc_m = BinaryAccuracy()
    auc_m = AUC()

    f1_score = F1_score(outs.to(torch.float32),gts.to(torch.float32))
    precision = precision_m(outs.to(torch.float32),gts.to(torch.float32))
    recall = recall_m(outs.to(torch.float32),gts.to(torch.float32))

    acc=[]
    auc=[]
    
    for i in range(len(outs)):
        out = outs[i]
        gt = gts[i]
        
        # Binary Accuracy
        bAcc_m.update(out, gt)
        ba = bAcc_m.compute()
        acc.append(ba.item())
        
        # AUC
        auc_m.update(out, gt)
        auc_val = auc_m.compute()
        auc.append(auc_val.item())
        
    acc = np.array(acc)
    acc = np.mean(acc)
    
    auc = np.array(auc)
    auc = np.mean(auc)
    
    return acc, f1_score.item(), precision.item(), recall.item(), auc   

def acc_each(outs,gts,num_classes=8):
    outs = np.array(outs)
    gts=  np.array(gts)

    true_list=np.zeros(num_classes)
    ct_list=np.ones(num_classes)*len(outs)

    for i in range(len(outs)):
        out = outs[i]
        gt = gts[i]

        for j in range(num_classes):
            if out[j]<0.5 and gt[j]==0:
                true_list[j]+=1
            elif out[j]>=0.5 and gt[j]==1:
                true_list[j]+=1

    acc_list = true_list/ct_list
    return acc_list

def get_false(outs,gts,paths,attribute_names,num_classes=8):
    outs = np.array(outs)
    gts=  np.array(gts)
    false_list={}
    for k in attribute_names:
        false_list[k]=[]
    
    for i in range(len(outs)):
        out = outs[i]
        gt = gts[i]
        for j in range(num_classes):
            if out[j]<0.5 and gt[j]==1:
                false_list[attribute_names[j]].append([out[j],paths[i]])
            elif out[j]>=0.5 and gt[j]==0:
                false_list[attribute_names[j]].append([out[j],paths[i]])

    return false_list

        

def get_args():
    parser = argparse.ArgumentParser(description='Test Attribute Models')
    parser.add_argument('--cfg_path', type=str,default='configs/efficientNet_B0_celebA.py')
    parser.add_argument('--model_path',type=str,default='')
    parser.add_argument('--test_format',type=str,default='h5',help='h5, csv ...')
    parser.add_argument('--test_base',type=str,default='')
    parser.add_argument('--image_base',type=str,default="celebA/Img/img_celeba/")
    parser.add_argument('--need_align',action='store_true',default=False)
    parser.add_argument('--detection_path',type=str,default="scrfd_10g_bnkps.onnx")
    parser.add_argument('--false_save',action='store_true',default=False)
    parser.add_argument('--result_save',action='store_true',default=False)

    args = parser.parse_args()

    return args

def test_csv(model,cfg,args,detection_model=None,ref_land=None,get_paths=False):
    base = args.test_base
    image_base = args.image_base

    df = pd.read_csv(base)
    attribute_names = cfg.attribute_names

    nt=0

    outs=[]
    gts=[]
    if get_paths:
        paths=[]

    for val in tqdm(df.values):
        img_path = os.path.join(image_base,val[0])
        gt_list=[val[-1],*val[4:-1]]
        if not os.path.exists(img_path):
            nt+=1
            continue
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

        if args.need_align:
            faces = get_detection('scrfd',detection_model,img,thresh=0.5,height_min=0,input_size=(640,640))
            if len(faces)<1:
                print("not detected")
            elif len(faces)==1:
                face = faces[0]
            else:
                for face in faces:
                    if face.max_flag:
                        break
                        
            land = face.land5
            img = align(img,land,ref_land,cfg.input_size)

        # preprocessing
        input_img = preprocessing(img)

        # forward
        out = model(input_img)
        out = out[0].detach().numpy()

        outs.append(out)
        gts.append(gt_list)
        paths.append(img_path)

    outs = torch.tensor(outs)
    gts = torch.tensor(gts)

    if get_paths:
        return gts,outs,len(df)-nt,paths
    else:
        return gts,outs,len(df)-nt

def test_h5(model,cfg,args,detection_model=None,ref_land=None,get_paths=False):
    base = args.test_base
    
    hf = h5py.File(base,'r')
    images = hf['images']
    labels = hf['labels']

    attribute_names = cfg.attribute_names

    outs=[]
    gts=[]
    if get_paths:
        paths=[]

    for i in tqdm(range(len(images))):
        gt_list=labels[i]
        img = images[i]

        if args.need_align:
            faces = get_detection('scrfd',detection_model,img,thresh=0.5,height_min=0,input_size=(640,640))
            if len(faces)<1:
                print("not detected")
            elif len(faces)==1:
                face = faces[0]
            else:
                for face in faces:
                    if face.max_flag:
                        break
                        
            land = face.land5
            img = align(img,land,ref_land,cfg.input_size)

        # preprocessing
        input_img = preprocessing(img)

        # forward
        out = model(input_img)
        out = out[0].detach().numpy()

        outs.append(out)
        gts.append(gt_list)
        paths.append(i)

    outs = torch.tensor(outs)
    gts = torch.tensor(gts)

    if get_paths:
        return gts,outs,len(images),paths
    else:
        return gts,outs,len(images)


def main():
    args = get_args()
    cfg_path = args.cfg_path
    cfg = get_config(cfg_path)

    # if need align
    if args.need_align:
        print("Need Align")
        if 'add' in cfg_path:
            ori_size=112
            new_size=cfg.input_size
            add_size=(new_size-ori_size)//2
            ref_land = LMARK_REF_ARC+[add_size,add_size]
        else:
            if cfg.input_size==224:
                ref_land = LMARK_REF_ARC*2
            elif cfg.input_size==112:
                ref_land = LMARK_REF_ARC

        # load detection model
        detection_path = args.detection_path
        detection_name = "scrfd"
        # Set parameters
        detection_thresh = 0.5
        detection_height_min=0 
        detection_model = get_detection_model(detection_name,detection_path,load_multi=False)
    else:
        detection_model=None
        ref_land=None

        

    # load model
    model_path = args.model_path
    if os.path.isdir(model_path):
        best_list=[]
        for filename in os.listdir(model_path):
            if "best" in filename:
                best_list.append(filename)
        model_path = os.path.join(model_path,sorted(best_list)[-1])


    model = build_model(cfg.network,cfg.num_classes,'',False)
    load_weight = torch.load(model_path)
    new_state_dict = OrderedDict()
    for n, v in load_weight.items():
        name = n.replace("module.","") 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    _ = model.eval()

    start_run=time.time()
    if args.test_format=='csv':
        results = test_csv(model,cfg,args,detection_model,ref_land,get_paths=args.false_save)
        if args.false_save:
            gts,outs,total,paths = results
        else:
            gts,outs,total = results
    elif args.test_format=='h5':
        results = test_h5(model,cfg,args,detection_model,ref_land,get_paths=args.false_save)
        if args.false_save:
            gts,outs,total,paths = results
        else:
            gts,outs,total = results
    else:
        print("test format wrong...exit..!")
        exit()
    end_run=time.time()

    if args.result_save:
        result_path = os.path.join(os.path.dirname(model_path),'pred_list.txt')
        olist = np.array(outs)
        glist = np.array(gts)

        with open(result_path,'w') as f:
            for i in range(len(olist)):
                f.writelines("{},{}\n".format(olist[i],glist[i]))

    print("Evaluation...")
    start_eval=time.time()
    acc, f1_score, precision, recall, auc = eval(outs,gts)
    acc_list = acc_each(outs,gts,cfg.num_classes)
    end_eval=time.time()
    print("Finish!")

    if args.false_save:
        false_list = get_false(outs,gts,paths,cfg.attribute_names,num_classes=8)
        save_path = os.path.join(os.path.dirname(model_path),'false_dict.npy')
        np.save(save_path,false_list)



    save_txt_path = os.path.join(os.path.dirname(model_path),'result.txt')

    with open(save_txt_path,'w') as f:
        line = "model: {}".format(args.model_path)
        print(line)
        f.writelines(line+"\n")

        line = "test: {}".format(args.test_base)
        print(line)
        f.writelines(line+"\n")

        line = "run time: {} ms".format(((end_run-start_run)*1000/total))
        print(line)
        f.writelines(line+"\n")

        line = "eval time: {} ms".format(((end_eval-start_eval)*1000/total))
        print(line)
        f.writelines(line+"\n")

        print("< Results >...")
        print("  Acc:",round(acc*100,3),"%")
        print("  F1-score:",f1_score)
        print("  Precision:",precision)
        print("  Recall:",recall)
        print("  AUC:",auc)

        line = "< Result >...\n  Acc: {} %\n  F1-score: {}\n  Precision: {}\n  Recall: {}\n  AUC: {}\n".format(round(acc*100,3),f1_score,precision,recall,auc)
        f.writelines(line+"\n")

        print("< Each Acc >...")
        f.writelines("< Each Acc >...\n")
        for ki,k in enumerate(cfg.attribute_names):
            line = "{} : {}%".format(k,acc_list[ki])
            print(line)
            f.writelines(line+"\n")


if __name__ == "__main__":
    main()