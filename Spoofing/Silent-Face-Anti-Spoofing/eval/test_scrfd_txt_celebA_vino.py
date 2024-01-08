
import os
import cv2
import numpy as np
import argparse
import warnings
import time
import torch
from tqdm import tqdm

from src.anti_spoof_predict_scrfd import Detection
from src.generate_patches import CropImage

from totalface.model_zoo.model_common import load_openvino

import random

warnings.filterwarnings('ignore')

import time
from datetime import datetime
import json


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

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def test(args):
    
    image_txt,image_base, model_path_,norm_input = args.image_txt, args.image_base, args.model_path, args.norm_input
    scale,save_base,save_name = args.scale, args.save_base,args.save_name

    if norm_input:
        print("need norm code")
        exit()

    if scale=='org':
        scale = None
        w_input=80
        h_input=60
    else:
        scale = float(scale)
        w_input=80
        h_input=80

    model_path = [model_path_.split(".vino")[0]+".xml",model_path_.split(".vino")[0]+".bin"]
    model = load_openvino.Openvino(model_path = model_path,not_norm=True,torch_image=True)


    save_dir_name = save_name.split("_test")[0]
    save_root = os.path.join(save_base,save_dir_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    score_save = os.path.join(save_root,save_name+"_scores.npy")
    txt_save = os.path.join(save_root,save_name+"_result.txt")

    detector = Detection()
    image_cropper = CropImage()

    lines = open(image_txt,'r').readlines()

    label_list = ['real','print','replay','3d']

    box_times=0
    times=0
    pretimes=0
    gt=[]
    pred=[]
    save_scores=[]
    real_scores=[]
    
    for line in tqdm(lines):
        sp = line.strip().split(" ")
        img_path = os.path.join(image_base,sp[0])
        label_str = int(sp[1])
    
        box_s = time.time()
        image = cv2.imread(img_path)
        image_bbox = detector.get_bbox(image)
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
        img = img.astype(np.float)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        pre_end = time.time()
        pretimes += ((pre_end-pre_st)*1000)

        st = time.time()
        prediction_,ft_map = model(img)
        prediction = softmax(prediction_)
        ed = time.time()
        times +=((ed-st)*1000)

        label_pred = np.argmax(prediction)

        if label_str==0:
            label_gt=1 #real True 1
        else:
            label_gt=0

        save_scores.append([prediction[0],int(label_str)]) # real 0 / spoof 1
        real_scores.append([prediction[0][0]])

        gt.append(label_gt)

        if label_pred==0:
            pred.append(1)
        else:
            pred.append(0)

    total = len(pred)

    with open(txt_save,'w') as f:
        line = ">> detection times: {} ms".format((box_times/total))
        f.writelines(line+"\n")
        print(line)

        line = ">> preprocessing times: {} ms".format((pretimes/total))
        f.writelines(line+"\n")
        print(line)

        line = ">> infer times: {} ms".format((times/total))
        f.writelines(line+"\n")
        print(line)

        print("celebA test results ... ")
        f.writelines("< Test Result >\n")

        tn,tp,fn,fp,tpr,fpr,apcer,bpcer,acer,acc = eval(gt,pred)
        frr,far,eer,hter = eval_eer(gt,real_scores)
        print(">> ACC:",acc)
        print(">> APCER:",apcer)
        print(">> BPCER:",bpcer)
        print(">> ACER:",acer)
        print(">> HTER:",hter)
        print(">> EER:",eer)
        f.writelines("* ACC: {} \n".format(acc))
        f.writelines("* APCER: {} \n".format(apcer))
        f.writelines("* BPCER: {} \n".format(bpcer))
        f.writelines("* ACER: {} \n".format(acer))
        f.writelines("* HTER: {} \n".format(hter))
        f.writelines("* EER: {} \n".format(eer))
    if args.np_save:
        np.save(score_save.format("all"),np.array(save_scores))
    

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--model_path",
        type=str,
        default="~.vino",
        help="model path list")
    parser.add_argument(
        "--image_txt",
        type=str,
        default="CelebA_Spoof/metas/intra_test/test_label.txt",
        help="image txt path used to test")
    parser.add_argument("--image_base",type=str,default="CelebA_Spoof/")
    parser.add_argument('--norm_input',action='store_true',default=False)
    parser.add_argument("--scale",type=str,default='2.7',help=['org','1','2.7','4'])
    parser.add_argument("--save_base",type=str,default="./results_test/")
    parser.add_argument("--save_name",type=str,default="")
    parser.add_argument("--np_save",action='store_true',default=False)

    args = parser.parse_args()
    test(args)
