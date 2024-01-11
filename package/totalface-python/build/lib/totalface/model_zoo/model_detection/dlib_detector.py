import numpy as np
import cv2
import os
import sys

import dlib

from ...data.image import read_image
from ...utils.util_detection import get_dlib_result



class Dlib:
    def __init__(self,shape_file):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_file)

    def detect(self,img,thresh):
        det, scores, lands = get_dlib_result(img,self.detector,self.predictor,thresh)

        lands5=[]
        for land in lands:
            eye_left = [land[37][0]+(land[38][0]-land[37][0])/2, land[41][1]-(land[41][1]-land[37][1])/2]
            eye_right = [land[43][0]+(land[44][0]-land[43][0])/2, land[47][1]-(land[47][1]-land[43][1])/2]
            nose = list(land[30])
            mouth_left = [land[48][0]+(land[60][0]-land[48][0])/2, land[60][1]-(land[48][1]-land[60][1])/2]
            mouth_right = [land[64][0]+(land[54][0]-land[64][0])/2, land[64][1]-(land[54][1]-land[64][1])/2]
            
            lands5.append([eye_left,eye_right,nose,mouth_left,mouth_right])
        lands5 = np.array(lands5,dtype=np.float32)

        return det, scores, lands5

