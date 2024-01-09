#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

# original code
# def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
#     im = np.ascontiguousarray(np.copy(image))
#     im_h, im_w = im.shape[:2]

#     top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

#     #text_scale = max(1, image.shape[1] / 1600.)
#     #text_thickness = 2
#     #line_thickness = max(1, int(image.shape[1] / 500.))
#     text_scale = 2
#     text_thickness = 2
#     line_thickness = 3

#     radius = max(5, int(im_w/140.))
#     cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
#                 (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

#     for i, tlwh in enumerate(tlwhs):
#         x1, y1, w, h = tlwh
#         intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
#         obj_id = int(obj_ids[i])
#         id_text = '{}'.format(int(obj_id))
#         if ids2 is not None:
#             id_text = id_text + ', {}'.format(int(ids2[i]))
#         color = get_color(abs(obj_id))
#         cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
#         cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
#                     thickness=text_thickness)
#     return im

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, \
                       draw_info=True,draw_box=True,draw_id=True,draw_blur=False,min_size=15):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    
    if draw_blur:
        maskShape = (im.shape[0], im.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)
        tempImg = im.copy()

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    if draw_info:
        cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        if w<min_size or h<min_size:
            continue
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        if draw_box:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        if draw_id:
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
        if draw_blur:
            int_xywh = tuple(map(int, (x1, y1, w, h)))
            x,y,w,h = int_xywh
            if x<0:
                x=0
            if y<0:
                y=0
            if x>im.shape[1]:
                x = im.shape[1]-1
            if h>im.shape[0]:
                h = im.shape[0]-1
            tempImg[y:y+h, x:x+w] = cv2.blur(tempImg[y:y+h, x:x+w] ,(23,23))
            
            center = (x+(w//2),y+(h//2))
            axesLength = (round(w/2),round(h*0.575))
            angle=0
            startAngle=0
            endAngle=360
            color = (0,255,0)
            thickness=3
            
            cv2.ellipse(mask , center ,axesLength, angle,startAngle,endAngle,255,-1)
            
    if draw_blur:
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(im,im,mask = mask_inv)
        img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
        im = cv2.add(img1_bg,img2_fg) 
            
    return im

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
