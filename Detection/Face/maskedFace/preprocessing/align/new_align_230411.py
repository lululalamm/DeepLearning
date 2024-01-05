import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from totalface.data.image import read_image
from totalface.model_zoo.get_models import get_detection_model,get_mask_model
from totalface.face.get_result import get_detection,get_mask
from totalface.utils.util_warp import face_align
from totalface.data.constant import LMARK_REF_ARC


# detection

detection_name = "scrfd"
detection_path = "scrfd_10g_bnkps.onnx"

# Set parameters
detection_thresh = 0.5
detection_height_min=0 

detection_model = get_detection_model(detection_name,detection_path,load_multi=False)


df_format = "FaceMask/filtering/saved_{}_id_balanced.csv"
save_base = "FaceMask/id_balanced/images/"

ori_size=112
new_size=160
add_size=(new_size-ori_size)//2

for tv in ['train','val']:
    df = pd.read_csv(df_format.format(tv))
    print("Start",tv)
    for val in tqdm(df.values):
        ori_id, ori_path, gender, gs,mask,new_id, new_name, mask_type = val
        ori_path = ori_path.replace("notebook/","/")
        save_path = os.path.join(save_base,new_name)

        img = read_image(ori_path)

        faces = get_detection(detection_name,detection_model,img,thresh=detection_thresh,height_min=detection_height_min,input_size=(640,640))

        LMARK_REF_ARC_new = LMARK_REF_ARC+[add_size,add_size]
        aimg = face_align(img,LMARK_REF_ARC_new,faces[0].land5,new_size)
        aimg = cv2.cvtColor(aimg,cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path,aimg)