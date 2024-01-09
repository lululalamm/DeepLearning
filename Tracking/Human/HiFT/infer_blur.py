from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.model_load import load_pretrain

def get_blur(frame,x1,y1,x2,y2,ksize=81):
    dimg = frame.copy()
    roi = frame[y1:y2, x1:x2]
    blur_image = cv2.GaussianBlur(roi,(ksize,ksize),0)
    dimg[y1:y2,x1:x2] = blur_image
    return dimg

# HiFT load
config_path = "./experiments/config.yaml"
snapshot = "./pretrained_models/general_model.pth"

# config
cfg.merge_from_file(config_path)
cfg.CUDA = torch.cuda.is_available()
device = torch.device('cuda' if cfg.CUDA else 'cpu')

# load model
model = ModelBuilder()
model = load_pretrain(model, snapshot).eval().to(device)

# build tracker
tracker = HiFTTracker(model)

# video load
vid_path = "test_w960.mp4"
save_path="result_HiFT_test_blur.mp4"
is_blur=True

cap = cv2.VideoCapture(vid_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if save_path:
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

# window init
cv2.namedWindow("Tracking_HiFT")#,cv2.WND_PROP_FULLSCREEN)
new_width = 600
if new_width!=-1:
    w_scale = width/new_width
    new_height = int(height//w_scale)
    cv2.resizeWindow('Tracking_HiFT', width=new_width, height=new_height)

# inference
start=True
c=1
while True:
    
    ret_val,frame = cap.read()

    if start:
        cv2.imshow('Tracking_HiFT',frame)

    if start and cv2.waitKey()==ord('p'):
        print("skip frame")
        continue
    
    elif start and cv2.waitKey()==ord('s'):
        print("Start")



    if ret_val:
        if start: # init
            init_rect = cv2.selectROI(windowName='Tracking_HiFT',img=frame, showCrosshair=False)
            print("init rect:",init_rect)
            tracker.init(frame,init_rect)
            if save_path:
                if is_blur:
                    frame_copy = frame.copy()
                    frame_copy = get_blur(frame_copy,init_rect[0],init_rect[1],init_rect[0]+init_rect[2],init_rect[1]+init_rect[3])
                    vid_writer.write(frame_copy)
                else:
                    frame = cv2.rectangle(frame,(init_rect[0],init_rect[1]),(init_rect[0]+init_rect[2],init_rect[1]+init_rect[3]),(0,255,0),3)
                    vid_writer.write(frame)
            start=False

        else: # tracking
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            if is_blur:
                frame_copy = frame.copy()
                frame_copy = get_blur(frame_copy,bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                            (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                            (0, 255, 0), 3)
            if save_path:
                if is_blur:
                    vid_writer.write(frame_copy)
                else:
                    vid_writer.write(frame)
            if new_width!=-1:
                frame = cv2.resize(frame,(new_width,new_height))
            cv2.imshow('Tracking_HiFT',frame)
            

    else: 
        break

    # keyboard interrupted  
    k = cv2.waitKey()
    if k==27: # esc
        break

    if k==ord('r'):
        print("restart")
        if save_path:
            vid_writer.release()
            vid_writer = cv2.VideoWriter(
                save_path.split(".mp4")[0]+"-{}.mp4".format(c), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
            )
            
            c+=1
        start = True

    
cap.release()
if save_path:
    vid_writer.release()
cv2.destroyAllWindows()
