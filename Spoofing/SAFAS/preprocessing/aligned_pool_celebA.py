import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse

import multiprocessing

from totalface_cpu.model_zoo.get_models import get_detection_model
from totalface_cpu.face.get_result import get_detection
from totalface_cpu.data.image import read_image
from totalface_cpu.utils.util_warp import face_align
from totalface_cpu.data.constant import LMARK_REF_ARC

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",default="CelebA_Spoof/")
    parser.add_argument('--txt_format', default="CelebA_Spoof/metas/intra_test/{}_label.txt")
    parser.add_argument('--save_base',default="./aligned_256/")
    parser.add_argument('--imgsize',default=256)
    parser.add_argument('--dt_name',default='scrfd')
    parser.add_argument('--dt_path',default="scrfd_10g_bnkps.onnx")
    parser.add_argument('--tv',default='test')
    args = parser.parse_args()
    return args

# global
args = get_args()
base = args.base
txt_format = args.txt_format
tv = args.tv
save_base = args.save_base
save_base = os.path.join(save_base,tv)
imgsize = args.imgsize

# load detection model
detection_name = args.dt_name
detection_path = args.dt_path

# Set parameters
detection_thresh = 0.5
detection_height_min=0 

# if load multiple trt, load_multi=True, default False
detection_model = get_detection_model(detection_name,detection_path,load_multi=False)
scale = imgsize/112

def update(*a):
    pbar.update()

def processing(img_path,lb_gt,total_idx):

    img = read_image(img_path)

    faces = get_detection(detection_name,detection_model,img,thresh=detection_thresh,height_min=detection_height_min,input_size=(640,640))
    if len(faces)<1:
        return

    elif len(faces)>1:
        for face in faces:
            if face.max_flag:
                break
    else:
        face = faces[0]

    LMARK_REF_ARC_256 = LMARK_REF_ARC*scale
    aimg = face_align(img,LMARK_REF_ARC_256,face['land5'],out_size=imgsize)

    aimg = cv2.cvtColor(aimg,cv2.COLOR_RGB2BGR)
    
    img_name = img_path.split("/")[-1]
    img_sub = os.path.join(*img_path.split("/")[:-1])
    if lb_gt==1:
        gt_str='real'
    else:
        gt_str='fake'
    img_format = img_name.split(".")[1]
    new_name = img_name.split(".")[0]+"_{}."#+img_name.split(".")[1]
    new_name = new_name.format(gt_str)[:-1]
    id_val = img_sub.split("/")[-2]

    new_name = "{}_{}_{}_{}.{}".format(tv,id_val,new_name,total_idx,img_format) # tv_id_imgname_spoof_total_idx.png jpg

    new_path = os.path.join(save_base,new_name)

    cv2.imwrite(new_path,aimg)

if not os.path.exists(save_base):
    os.makedirs(save_base)

txt_path = txt_format.format(args.tv)
lines = open(txt_path,'r').readlines()
img_list = []
label_gt_list=[]

for line in tqdm(lines):
    sp = line.strip().split(" ")
    img_path = os.path.join(base,sp[0])
    label_str = int(sp[1]) # real 0 / fake 1
    
    img_list.append(img_path)
    if label_str==0: # real
        label_gt=1
    else:
        label_gt=0
        
    label_gt_list.append(label_gt)

try:
    #save_list_f = open(save_list,'w')

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    pbar = tqdm(total=len(img_list))
    for i in range(pbar.total):
        pool.apply_async(processing, args=(img_list[i],label_gt_list[i],i), callback=update)
        #res.get()

    pool.close()
    pool.join()

except KeyboardInterrupt:
    pool.close()
    exit()

