import os
import numpy as np
import cv2
import onnxruntime

from ...utils.util_landmark import parse_roi_box_from_bbox, crop_img, recon_ver, calc_pose, get_distance_point
from ...data.constant import PARAM_STD, PARAM_MEAN, U_BASE, W_SHP_BASE, W_EXP_BASE, STD_SIZE

from ..model_common import load_onnx, load_openvino


class TDDFA3D_V2:
    def __init__(self,model_type,model_path,**kwargs):
        self.std_size = STD_SIZE
        self.param_std = PARAM_STD
        self.param_mean = PARAM_MEAN
        self.u_base = U_BASE
        self.w_exp_base = W_EXP_BASE
        self.w_shp_base = W_SHP_BASE
        self.dense = False
        self.model_path = model_path
        self.model_type = model_type
        self.left_eye_idx = [43,44,43,47]
        self.right_eye_idx = [41,40,38,40]

        if self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,output_sort=True,onnx_device='cpu')
        elif self.model_type=='openvino':
            self.net = load_openvino.Openvino(self.model_path,not_norm=True,device='CPU')

    def get(self,img,face,eye_dist=False):
        box = face.bbox
        roi_box = parse_roi_box_from_bbox(box)

        img = crop_img(img, roi_box)
        img = cv2.resize(img, dsize=(self.std_size,self.std_size), interpolation=cv2.INTER_LINEAR)
        img = (img - 127.5) / 128.
        # (112,112,3) normalized image
        
        param = self.net(img)
        if isinstance(param,list) and len(param)==1:
            param = param[0]

        param = param.flatten().astype(np.float32)
        param = param * self.param_std + self.param_mean  # re-scale

        P,pose = calc_pose(param)
        ver = recon_ver(param,roi_box,self.dense, self.std_size)

        preds_pre=[]
        for i in range(68):
            x_pred = ver[0][i]
            y_pred = ver[1][i]
            
            preds_pre.append([x_pred,y_pred])

        preds_pre = np.array(preds_pre).astype(np.float32)
        
        land5=[]
        eye_left = (ver[0][41]+(ver[0][40]-ver[0][41])/2 , ver[1][38]+(ver[1][40]-ver[1][38])/2)
        eye_right = (ver[0][43]+(ver[0][44]-ver[0][43])/2, ver[1][43]+(ver[1][47]-ver[1][43])/2)
        nose = (ver[0][30], ver[1][30])
        mouth_left = (ver[0][48]+(ver[0][60]-ver[0][48])/2 , ver[1][60]-(ver[1][48]-ver[1][60])/2)
        mouth_right = (ver[0][64]+(ver[0][54]-ver[0][64])/2 , ver[1][64]-(ver[1][54]-ver[1][64])/2)
        land5.append(eye_left)
        land5.append(eye_right)
        land5.append(nose)
        land5.append(mouth_left)
        land5.append(mouth_right)
        land5 = np.array(land5).astype(np.float32)
        
        face.land = preds_pre
        face.land5 = land5
        face.pose = pose

        if eye_dist:
            face.eye_dist=get_distance_point(eye_left,eye_right)
            
        return param, roi_box,face.land5
        
        

    def get_results(self,img_ori, faces):
        param_lst = []
        roi_box_lst = []
        
        boxes = [face.bbox for face in faces]

        for box in boxes:
            roi_box = parse_roi_box_from_bbox(box)
            roi_box_lst.append(roi_box)
        
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.std_size,self.std_size), interpolation=cv2.INTER_LINEAR)
            img = (img - 127.5) / 128.
            # (112,112,3) normalized image

            param = self.net(img)
            if isinstance(param,list) and len(param)==1:
                param = param[0]

            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)
        
        return param_lst, roi_box_lst


    def get_fromImage(self,img,bbox):
        roi_box = parse_roi_box_from_bbox(bbox)

        img = crop_img(img, roi_box)
        img = cv2.resize(img, dsize=(self.std_size,self.std_size), interpolation=cv2.INTER_LINEAR)
        img = (img - 127.5) / 128.
        # (112,112,3) normalized image
        
        param = self.net(img)
        if isinstance(param,list) and len(param)==1:
            param = param[0]
        param = param.flatten().astype(np.float32)
        param = param * self.param_std + self.param_mean  # re-scale

        P,pose = calc_pose(param)
        ver = recon_ver(param,roi_box,self.dense, self.std_size)

        preds_pre=[]
        for i in range(68):
            x_pred = ver[0][i]
            y_pred = ver[1][i]
            
            preds_pre.append([x_pred,y_pred])

        preds_pre = np.array(preds_pre).astype(np.float32)
        
        land5=[]
        eye_left = (ver[0][37]+(ver[0][38]-ver[0][37])/2 , ver[1][41]-(ver[1][41]-ver[1][37])/2)
        eye_right = (ver[0][43]+(ver[0][44]-ver[0][43])/2, ver[1][47]-(ver[1][47]-ver[1][43])/2)
        nose = (ver[0][30], ver[1][30])
        mouth_left = (ver[0][48]+(ver[0][60]-ver[0][48])/2 , ver[1][60]-(ver[1][48]-ver[1][60])/2)
        mouth_right = (ver[0][64]+(ver[0][54]-ver[0][64])/2 , ver[1][64]-(ver[1][54]-ver[1][64])/2)
        land5.append(eye_left)
        land5.append(eye_right)
        land5.append(nose)
        land5.append(mouth_left)
        land5.append(mouth_right)
        land5 = np.array(land5).astype(np.float32)
        
        return param, roi_box, preds_pre, land5, pose


