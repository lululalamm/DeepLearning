# inference example code
```python
import os
import numpy as np
import cv2

from nsface.model_zoo.get_models import get_detection_model,get_recognition_model
from nsface.face.get_result import get_detection
from nsface.data import read_image

from nsface.face.blur import face_blur_image,load_tddfa

# load models
# detection model
detection_name = "scrfd"
detection_path = ["detection/scrfd_10g_bnkps_quantINT8.xml",
                 "detection/scrfd_10g_bnkps_quantINT8.bin"]

detection_thresh = 0.5
detection_height_min=0 

detection_model = get_detection_model(detection_name,detection_path,load_multi=False)

# recog model
recog_name = 'arcface'
recog_path = ["embedding/mbf_arcface_20220210_quantINT8.xml",
             "embedding/mbf_arcface_20220210_quantINT8.bin"]
recog_out_size=112
recog_num_features=512
recog_network='mbf'
recog_fp16=True

recog_model = get_recognition_model(recog_name,recog_path,out_size=recog_out_size,num_features=recog_num_features,network=recog_network,fp16=recog_fp16)

# load tddfa
path = "landmark/3ddfa_v2_quantINT8"
tdd_format='vino'

tddfa_model = load_tddfa(path,tdd_format)

# load image
frame_path = "./test_data/frame_540.jpg"
img = read_image(frame_path)
ref_path = "./test_data/test_ref.jpeg"
ref_img = read_image(ref_path)

# blur image
# ref_img, recog_model 을 넣어 score_th=1.3(default) 미만 detect 들은 blur 처리
blur_image = face_blur_image(img,detection_name,detection_model,tddfa_model,recog_model,ref_img) 

# ref_img, recog_model 넣지 않으면 모든 detect 를 blur 처리
blur_image2 = face_blur_image(img,detection_name,detection_model,tddfa_model)