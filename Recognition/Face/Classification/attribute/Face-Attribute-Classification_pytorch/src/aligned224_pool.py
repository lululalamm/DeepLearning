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
LMARK_REF_ARC224 = LMARK_REF_ARC*2


def save_align(img_path):
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
        
    aimg = face_align(img,LMARK_REF_ARC224,face.land5,224)
    aimg = cv2.cvtColor(aimg,cv2.COLOR_RGB2BGR)

    save_path = os.path.join(save_base,img_path.split("/")[-1])
    cv2.imwrite(save_path,aimg)

def update(*a):
    pbar.update()


image_base = "celebA/Img/img_celeba/"
save_base = "img_align224_celeba_scrfd/"
if not os.path.exists(save_base):
    os.mkdir(save_base)
csv_path = "celebA/csv/230720/check_celebA.csv"
df = pd.read_csv(csv_path)


print("Start process")
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

pbar = tqdm(total=len(df))
for i in range(pbar.total):
    img_path = os.path.join(image_base,df.values[i][0])
    pool.apply_async(save_align, args=(img_path,), callback=update)

pool.close()
pool.join()

