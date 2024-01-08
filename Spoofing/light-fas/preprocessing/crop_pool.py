import os
import numpy as np
from math import ceil
import cv2
import argparse
from tqdm import tqdm

from totalface_cpu.model_zoo.get_models import get_detection_model
from totalface_cpu.face.get_result import get_detection
from totalface_cpu.data import read_image

import multiprocessing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",default="FaceInTheWild/{}/image_list_org_1_80x60_new_unmask.txt")
    parser.add_argument("--prefix",default="/data/")
    parser.add_argument("--tv",default='Training')
    parser.add_argument('--save_base',default="./crop_safas_wild_{}/")
    parser.add_argument('--dt_name',default='scrfd')
    parser.add_argument('--dt_path',default="scrfd_10g_bnkps.onnx")
    args = parser.parse_args()
    return args

def update(*a):
    pbar.update()

def processing(line,idx):

    sp = line.strip().split(",")
    ori_sub,_,_,_,id_val,age,gender,_,_,lb_str = sp

    ori_path = os.path.join(img_prefix,ori_sub)
    img_rgb = read_image(ori_path)

    faces = get_detection(detection_name,detection_model,img_rgb,thresh=detection_thresh,height_min=detection_height_min,input_size=(640,640))

    if len(faces)==1:
        face = faces[0]
    elif len(faces)<1:
        print("not detect")
        return
    else:
        for fc in faces:
            if fc['max_flag']:
                face = fc
                break

    box = face['bbox'].astype(np.int32)
    box = np.maximum(box, 0)

    cropped = img_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    cropped_bgr = cv2.cvtColor(cropped,cv2.COLOR_RGB2BGR)

    if lb_str=='real':
        lb_save = 'real'
    else:
        lb_save = 'fake'
    new_name = "{}_{}_{}.jpg".format(idx,id_val,lb_save)
    save_path = os.path.join(save_base,new_name)
    cv2.imwrite(save_path,cropped_bgr)

    return



args = get_args()

# load detection model
detection_name = args.dt_name
detection_path = args.dt_path

# Set parameters
detection_thresh = 0.5
detection_height_min=0 

# if load multiple trt, load_multi=True, default False
detection_model = get_detection_model(detection_name,detection_path,load_multi=False)

# init
img_prefix = args.prefix
tv = args.tv
save_tv={'Training':'train','Validation':'val'}
txt_base = args.base

save_base = args.save_base
save_base = save_base.format(save_tv[tv])
if not os.path.exists(save_base):
    os.makedirs(save_base)

lines = open(txt_base.format(tv),'r').readlines()

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
pbar = tqdm(total=len(lines))
for i in range(pbar.total):
    pool.apply_async(processing, args=(lines[i],i), callback=update)

pool.close()
pool.join()

