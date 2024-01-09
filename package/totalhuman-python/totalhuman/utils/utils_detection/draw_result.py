import os
import numpy as np
import cv2

def draw_box(dimg,bboxes,line_w=2,color=(0,255,0)):
    dimg = dimg.copy()
    for bbox in bboxes:
        bbox = bbox.astype(np.int32)
        dimg = cv2.rectangle(dimg,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,line_w)


    return dimg