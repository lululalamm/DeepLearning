import os
import numpy as np
import cv2
import skimage

from totalface.data.image import read_image
from totalface.data.constant import LMARK_REF_ARC
from totalface.model_zoo.get_models import get_detection_model
from totalface.face import get_detection

import mxnet as mx
from insightface.data.rec_builder import RecBuilder

from tqdm import tqdm

def face_align(img,lmark_ref,kps,out_size):

    st = skimage.transform.SimilarityTransform()
    st.estimate(kps, lmark_ref)
    M = st.params[0:2, :]
    
    aligned = cv2.warpAffine(img, M, (out_size,out_size),borderMode=cv2.BORDER_CONSTANT)

    return aligned



# load detection model
dpath = "scrfd_10g_bnkps.v8.trt"
dmodel = get_detection_model("scrfd",dpath)
out_size=112

# file paths
rec_save_dir = "./rec_files/"
#csv_path = "./FaceMask_Training_list_attr_new.csv"
csv_path = "./FaceMask_Training_list_attr_new_id.csv"

# rec builder
wrec = RecBuilder(rec_save_dir)

# load csv
csv_lines = open(csv_path,'r').readlines()

pass_ct=0
for line in tqdm(csv_lines):
    sp = line.strip().split(",")
    idname, img_path, gender, gw, mask, new_id = sp
    
    mask_type = int(img_path.split(".")[0].split("_")[-1])
    if mask_type==2:
        continue

    img = read_image(img_path)#,to_bgr=True)
    faces = get_detection("scrfd",dmodel,img,thresh=0.5,height_min=0)
    if len(faces)<1:
        continue
        pass_ct+=1
    elif len(faces)>1:
        for face in faces:
            if face['max_flag']:
                break
    else:
        face = faces[0]
        
    aligned = face_align(img,LMARK_REF_ARC,face.land5,out_size)
    aligned = cv2.cvtColor(aligned,cv2.COLOR_RGB2BGR) # rec must save bgr image
        
    #wrec.add_image(aligned,int(idname))
    wrec.add_image(aligned,int(new_id))

wrec.close()
print("pass ct:",pass_ct)
print("save directory:",rec_save_dir)