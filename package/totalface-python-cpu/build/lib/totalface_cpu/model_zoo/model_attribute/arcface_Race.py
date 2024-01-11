import cv2

import numpy as np

from ...data.image import read_torchImage
from ...utils.util_warp import face_align
from ...utils.util_attribute import get_pred
from ...data.constant import LMARK_REF_ARC

from ..model_common import load_onnx, load_openvino


label_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Others',5:'Others'}


class Arcface_Race:
    def __init__(self,model_type,model_path,out_size=112,**kwargs):
        self.model_path = model_path
        self.out_size=out_size
        self.model_type=model_type

        if self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,onnx_device='cpu')
        elif self.model_type=='openvino':
            self.net = load_openvino.Openvino(self.model_path,device='CPU')

    def get(self,img,face,to_bgr,mask_off=False,eye_min=0):
        if mask_off and np.argmax(face['mask_sf'])==1:
            face.race=[]
            return face.race
        if eye_min>0 and 'eye_dist' in face.keys() and face.eye_dist<eye_min:
            face.race=[]
            return face.race

        if not 'aimg' in face.keys():
            aimg = face_align(img,LMARK_REF_ARC,face.land5,self.out_size)
            face.aimg = aimg
        else:
            aimg = face.aimg

        if self.model_type=='onnx':
            aimg = (aimg/255. - 0.5)/0.5

        output = self.net(aimg)[0]
        idx = np.argsort(output)[0,::-1][0]
        p = output[0][idx]
        pred_name = label_dict[idx]

        face.race=pred_name

        return face.race







        

    









