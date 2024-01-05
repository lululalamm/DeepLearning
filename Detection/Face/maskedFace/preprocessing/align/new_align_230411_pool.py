import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing

from totalface_cpu.data.image import read_image
from totalface_cpu.model_zoo.get_models import get_detection_model
from totalface_cpu.face.get_result import get_detection
from totalface_cpu.utils.util_warp import face_align
from totalface_cpu.data.constant import LMARK_REF_ARC

# detection
detection_name = "scrfd"
detection_path = "scrfd_10g_bnkps.onnx"
# Set parameters
detection_thresh = 0.5
detection_height_min=0 
detection_model = get_detection_model(detection_name,detection_path,load_multi=False)

ori_size=112
new_size=160
add_size=(new_size-ori_size)//2


def save_align(img_path,save_path):
    img = read_image(img_path)
    faces = get_detection(detection_name,detection_model,img,thresh=detection_thresh,height_min=detection_height_min,input_size=(640,640))

    LMARK_REF_ARC_new = LMARK_REF_ARC+[add_size,add_size]
    aimg = face_align(img,LMARK_REF_ARC_new,faces[0].land5,new_size)
    aimg = cv2.cvtColor(aimg,cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path,aimg)

def update(*a):
    pbar.update()


df_format = "FaceMask/filtering/saved_{}_id_balanced.csv"
save_base = "aihub_dataset/FaceMask/id_balanced/images/"


image_paths=[]
save_paths=[]

print("Get paths")
#for tv in ['train','val']:
for tv in ['test']:
    df = pd.read_csv(df_format.format(tv))
    print("Start",tv)
    for val in tqdm(df.values):
        ori_id, ori_path, gender, gs,mask,new_id, new_name, mask_type = val
        ori_path = ori_path.replace("notebook/","/")
        save_path = os.path.join(save_base,new_name)

        image_paths.append(ori_path)
        save_paths.append(save_path)


print("Start process")
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

pbar = tqdm(total=len(image_paths))
for i in range(pbar.total):
    pool.apply_async(save_align, args=(image_paths[i],save_paths[i],), callback=update)

pool.close()
pool.join()