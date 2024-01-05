import os 
import numpy as np
from PIL import Image
import torch
import cv2
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import auc, roc_curve

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

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default='mobilenetv3-small',help="mnasnet / mobilenetv3-small / mobilefacenet")
    parser.add_argument("--model_path",type=str,default="./save_models/mobilenetv3-small_facemask_230227/last.pth")
    parser.add_argument("--num_classes",type=int,default=2)
    parser.add_argument("--test_base",type=str,default="rmfd_mfr2/mfr2_2_scrfd_aligned/")
    parser.add_argument("--save_base",type=str,default="./save_results/")
    parser.add_argument("--load_type",type=int,default=1)
    parser.add_argument("--input_size",type=int,default=112)
    parser.add_argument("--net_size",type=str,default='075',help="075 or 10")
    parser.add_argument("--save_name",type=str,default="")
    args = parser.parse_args()
    return args


lb_dict={0:'unmask',1:'mask'}
gt_dict={'unmask':0,'mask':1}

args = get_args()
model_name = args.model_name
model_path = args.model_path
num_classes = args.num_classes
test_base = args.test_base
save_base = args.save_base
if not os.path.exists(save_base):
    os.mkdir(save_base)

if not args.save_name:
    save_txt = os.path.join(save_base,model_path.split("/")[-2]+"_test_mfr2.txt")
    result_txt = os.path.join(save_base,model_path.split("/")[-2]+"_result_mfr2.txt")
else:
    save_txt = os.path.join(save_base,args.save_name+"_test_mfr2.txt")
    result_txt = os.path.join(save_base,args.save_name+"_result_mfr2.txt")

# load model
if model_name=='mnasnet':
    model = get_model(model_name,num_classes,load_type=args.load_type,mnas_size=args.net_size)
else:
    model = get_model(model_name,num_classes,load_type=args.load_type,input_size=args.input_size)

load_weight = torch.load(model_path)
new_state_dict = OrderedDict()
for n, v in load_weight.items():
    name = n.replace("module.","") 
    new_state_dict[name] = v

if model_name=='mnasnet':
    print("mnas load")
    if not hasattr(new_state_dict, '_metadata'):
         setattr(new_state_dict, '_metadata', OrderedDict({'version': 2}))


model.load_state_dict(new_state_dict)
_ = model.eval()

input_size = args.input_size


# start
gts=[]
preds=[]
mask_ct=0
mask_true=0
unmask_ct=0
unmask_true=0

all_time=0
time_c=0

with open(save_txt,'w') as f:
    for idname in tqdm(os.listdir(test_base)):
        if ".DS" in idname: continue
        for img_name in os.listdir(os.path.join(test_base,idname)):
            if ".DS" in img_name: continue

            img_path = os.path.join(test_base,idname,img_name)
            gt_str = img_name.split(".")[0].split("_")[-1]
            if gt_str=='unmask':
                unmask_ct+=1
            else:
                mask_ct+=1
            gts.append(gt_dict[gt_str])

            input_img = image_preprocess(img_path,resize=input_size)

            with torch.no_grad():
                s = time.time()
                output = model(input_img).numpy()
                e = time.time()
                output = softmax(output)
            lb = lb_dict[np.argmax(output)]
            preds.append(np.argmax(output))
            time_c+=1

            if gt_str=='unmask' and lb=='unmask':
                unmask_true+=1
            if gt_str=='mask' and lb=='mask':
                mask_true+=1

            new_line = "{},{},{},{},{}\n".format(img_path,gt_str,gt_dict[gt_str],lb,np.argmax(output))
            f.writelines(new_line)

            all_time+=((e-s)*1000)


# eval ( true - mask / false - unmask )

tn, fp, fn, tp = metrics.confusion_matrix(y_true=gts,y_pred=preds,).ravel()
acc = (tp+tn)/(tn+fp+fn+tp)
speed = (all_time/time_c)

# apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
# bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0

# acer = (apcer + bpcer) / 2

# fpr, tpr, threshold = roc_curve(gts, preds, pos_label=1)

# fnr = 1 - tpr

# eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

# fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
# fnr_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

# eer = (fpr_eer + fnr_eer) / 2

with open(result_txt,'w') as f:
    result_line = "TN : {} | FP : {} | FN : {} | TP : {}".format(tn,fp,fn,tp)
    print(result_line)
    f.writelines(result_line+"\n")

    result_line = "ACC: {}".format(acc)
    print(result_line)
    f.writelines(result_line+"\n")

    result_line = "Unmask Acc: {}".format((unmask_true/unmask_ct)*100)
    print(result_line)
    f.writelines(result_line+"\n")

    result_line = "Mask Acc: {}".format((mask_true/mask_ct)*100)
    print(result_line)
    f.writelines(result_line+"\n")

    result_line = "Speed: {} ms".format(speed)
    print(result_line)
    f.writelines(result_line+"\n")




