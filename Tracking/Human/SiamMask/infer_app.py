import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import logging
import matplotlib.pyplot as plt

import sys
sys.path.append("./experiments/siammask_sharp/")
from custom import Custom

sys.path.append("./")
from utils.load_helper import load_pretrain
from tools.test import siamese_init, siamese_track

import json

def proccess_loss(cfg):
    if 'reg' not in cfg:
        cfg['reg'] = {'loss': 'L1Loss'}
    else:
        if 'loss' not in cfg['reg']:
            cfg['reg']['loss'] = 'L1Loss'

    if 'cls' not in cfg:
        cfg['cls'] = {'split': True}

    cfg['weight'] = cfg.get('weight', [1, 1, 36])  # cls, reg, mask
    
def load_config(config_path,clip=None):
    config = json.load(open(config_path))

    # deal with network
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')

        config['network'] = {}

        if not args.arch:
            raise Exception('no arch provided')
    arch = config['network']['arch']

    # deal with loss
    if 'loss' not in config:
        config['loss'] = {}

    proccess_loss(config['loss'])

    # deal with lr
    if 'lr' not in config:
        config['lr'] = {}
    default = {
            'feature_lr_mult': 1.0,
            'rpn_lr_mult': 1.0,
            'mask_lr_mult': 1.0,
            'type': 'log',
            'start_lr': 0.03
            }
    default.update(config['lr'])
    config['lr'] = default

    # clip
    if clip:
        if 'clip' not in config:
            config['clip'] = {}
        config['clip'] = add_default(config['clip'],
                {'feature': clip, 'rpn': clip, 'split': False})
        if config['clip']['feature'] != config['clip']['rpn']:
            config['clip']['split'] = True
        if not config['clip']['split']:
            clip = config['clip']['feature']

    return config, arch,clip

# SiamMask load
config_path = "./experiments/siammask_sharp/config_davis.json"
resume = "./experiments/siammask_sharp/SiamMask_DAVIS.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

cfg = load_config(config_path)[0]

siammask = Custom(anchors=cfg['anchors'])
siammask = load_pretrain(siammask, resume)
_ = siammask.eval().to(device)

# video load
vid_path = "nba_clip.mp4"
save_path="./result_SiamMask_nba.mp4"

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
cv2.namedWindow("Tracking_SiamMask")
new_width = 600
if new_width!=-1:
    w_scale = width/new_width
    new_height = int(height//w_scale)
    cv2.resizeWindow('Tracking_SiamMask', width=new_width, height=new_height)

# inference
start=True
while True:
    ret_val,frame = cap.read()

    if ret_val:
        if start: # init
            x,y,w,h = cv2.selectROI(windowName='Tracking_SiamMask',img=frame, showCrosshair=False)
            if save_path:
                draw_init = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                vid_writer.write(frame)
            target_pos = np.array([x+w/2,y+h/2])
            target_sz = np.array([w,h])
            state = siamese_init(frame,target_pos, target_sz, siammask, cfg['hp'], device=device)
            start=False
        else: # tracking
            state = siamese_track(state, frame, mask_enable=True, refine_enable=True,device='cuda')
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
                
            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            if save_path:
                vid_writer.write(frame)
            if new_width!=-1:
                frame = cv2.resize(frame,(new_width,new_height))
            cv2.imshow('Tracking_SiamMask',frame)
            

    else: 
        break

    # keyboard interrupted  
    k = cv2.waitKey(1)
    if k==27: # esc
        break
cap.release()
if save_path:
    vid_writer.release()
cv2.destroyAllWindows()
