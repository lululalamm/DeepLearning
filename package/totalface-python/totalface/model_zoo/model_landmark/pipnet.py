import os
import numpy as np
import cv2
import torch
import onnxruntime

from ..model_common import load_tensorRT, load_onnx, load_openvino,load_torch
from ...utils.util_landmark import forward_pip_np, get_meanface, crop_pip, pip_get_5point
from ...data.constant import PIP_to_5

from ...data.experiments.COFW.pip_32_16_60_r18_l2_l1_10_1_nb10 import Config as config_cofw
from ...data.experiments.data_300W.pip_32_16_60_r18_l2_l1_10_1_nb10 import Config as config_300w
from ...data.experiments.WFLW.pip_32_16_60_r18_l2_l1_10_1_nb10 import Config as config_wflw
from ...data.experiments.WFLW.pip_32_16_60_mv3_large_l2_l1_10_1_nb10 import Config as config_wflw_mv3_large
from ...data.experiments.WFLW.pip_32_16_60_mv3_small_l2_l1_10_1_nb10 import Config as config_wflw_mv3_small
from ...data.experiments.AFLW.pip_32_16_60_r18_l2_l1_10_1_nb10 import Config as config_aflw

Config_dict={
    'cofw':["pip_32_16_60_r18_l2_l1_10_1_nb10","COFW",config_cofw()],
    '300w':["pip_32_16_60_r18_l2_l1_10_1_nb10","data_300W",config_300w()],
    'wflw':["pip_32_16_60_r18_l2_l1_10_1_nb10","WFLW",config_wflw()],
    'wflw_mv3-large':["pip_32_16_60_mv3_large_l2_l1_10_1_nb10","WFLW",config_wflw_mv3_large()],
    'wflw_mv3-small':["pip_32_16_60_mv3_small_l2_l1_10_1_nb10","WFLW",config_wflw_mv3_small()],
    'aflw':[ "pip_32_16_60_r18_l2_l1_10_1_nb10","AFLW",config_aflw()],
}


class PIPNet:
    def __init__(self,model_type,model_path,**kwargs):

        self.model_path = model_path
        self.model_type = model_type

        if isinstance(model_path,list):
            self.data_name = model_path[0].split("/")[-2]
            model_name = model_path[0].split("/")[-1]
        else:
            self.data_name = model_path.split("/")[-2]
            model_name = model_path.split("/")[-1]

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.data_name=='wflw':
            if "mv3" in model_name:
                if "small" in model_name:
                    self.data_name = 'wflw_mv3-small'
                else:
                    self.data_name = 'wflw_mv3-large'

        cfg_info = Config_dict[self.data_name]
        self.cfg = cfg_info[2]
        self.cfg.experiment_name = cfg_info[0]
        self.cfg.data_name = cfg_info[1]

        mean_base = kwargs.get("mean_base","/data/notebook/NAS/PTAS_Shared/resource/model/face/landmark/pipnet/meanfaces/")
        self.meanface_indices, self.reverse_index1, self.reverse_index2, self.max_len = get_meanface(os.path.join(mean_base, self.cfg.data_name, 'meanface.txt'), self.cfg.num_nb)
        
        # load model
        if self.model_type=='onnx':
            self.net = load_onnx.Onnx_session(self.model_path,input_mean=0.0, input_std=1.0,onnx_device=kwargs.get("onnx_device",'cuda'),torch_image=True)
        elif self.model_type=='trt':
            self.net = load_tensorRT.TrtModel(self.model_path,not_norm=True,torch_image=True)
            self.reshape_list = []
            for bind in self.net.engine:
                if "input" in bind: continue
                self.reshape_list.append(self.net.engine.get_binding_shape(bind))
        elif self.model_type=='openvino':
            output_order = ['outputs_cls', 'outputs_x', 'outputs_y', 'outputs_nb_x', 'outputs_nb_y']
            self.net = load_openvino.Openvino(self.model_path,not_norm=True,torch_image=True,outputs_order=output_order,device=kwargs.get("device",'CPU'))

    

    def get(self,img,face,input_size=256,scale_mul=1.1):
        bbox = face.bbox
        

        # norm N crop
        img, tmp_add, tmp_mul = crop_pip(img,bbox,target_size=input_size,scale_mul=scale_mul)

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        # torch input
        img = torch.tensor(img)
        img = torch.unsqueeze(img,dim=0)
        img_batch = img.shape[0]

        # inference
        output = self.net(img)
        if self.model_type=='trt':
            for i in range(len(output)):
                output[i] = np.reshape(output[i].ravel(),(img_batch,-1,8,8))

        # post process
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip_np(output,self.cfg.input_size, self.cfg.net_stride, self.cfg.num_nb)
        lms_pred = np.concatenate((lms_pred_x, lms_pred_y),axis=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
        tmp_x = np.mean(np.concatenate((lms_pred_x, tmp_nb_x), axis=1), axis=1).reshape(-1,1)
        tmp_y = np.mean(np.concatenate((lms_pred_y, tmp_nb_y), axis=1), axis=1).reshape(-1,1)
        lms_pred_merge = np.concatenate((tmp_x, tmp_y), axis=1).flatten()

        preds_pre=[]
        for j in range(self.cfg.num_lms):
            x_pred = lms_pred_merge[j*2]*tmp_mul[0] + tmp_add[0] # resize 인 경우 원래 이미지 크기 곱
            y_pred = lms_pred_merge[j*2+1]*tmp_mul[1] + tmp_add[1] # crop 돼서 포인트 기준이 바뀐 경우 x,y 바뀐만큼 더하기
            
            preds_pre.append([x_pred,y_pred])

        preds_pre = np.array(preds_pre)

        face.land = preds_pre
        land5 = pip_get_5point(self.cfg.num_lms,preds_pre,PIP_to_5)
        land5 = np.array(land5).astype(np.float32)
        face.land5 = land5
        
        return face.land5


    def get_results(self,img,faces,get_5point=False,input_size=256,scale_mul=1.1):
        land_list=[]
        if get_5point:
            land5_list=[]

        boxes = [face.bbox for face in faces]

        for bbox in boxes:
            # norm N crop
            img, tmp_add, tmp_mul = crop_pip(img,bbox,target_size=input_size,scale_mul=scale_mul)
            

            img = img.astype(np.float32)
            img = (img/255.0 - self.mean) / self.std
            img = img.transpose([2, 0, 1])

            # torch input
            img = torch.tensor(img)
            img = torch.unsqueeze(img,dim=0)
            img_batch = img.shape[0]

            # inference
            output = self.net(img)
            if self.model_type=='trt':
                for i in range(len(output)):
                    output[i] = np.reshape(output[i].ravel(),(img_batch,-1,8,8))

            # post process
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip_np(output,self.cfg.input_size, self.cfg.net_stride, self.cfg.num_nb)
            lms_pred = np.concatenate((lms_pred_x, lms_pred_y),axis=1).flatten()
            tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
            tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
            tmp_x = np.mean(np.concatenate((lms_pred_x, tmp_nb_x), axis=1), axis=1).reshape(-1,1)
            tmp_y = np.mean(np.concatenate((lms_pred_y, tmp_nb_y), axis=1), axis=1).reshape(-1,1)
            lms_pred_merge = np.concatenate((tmp_x, tmp_y), axis=1).flatten()

            preds_pre=[]
            for j in range(self.cfg.num_lms):
                x_pred = lms_pred_merge[j*2]*tmp_mul[0] + tmp_add[0] # resize 인 경우 원래 이미지 크기 곱
                y_pred = lms_pred_merge[j*2+1]*tmp_mul[1] + tmp_add[1] # crop 돼서 포인트 기준이 바뀐 경우 x,y 바뀐만큼 더하기
                
                preds_pre.append([x_pred,y_pred])

            preds_pre = np.array(preds_pre).astype(np.float32)
            land_list.append(preds_pre)

            if get_5point:
                land5 = pip_get_5point(self.cfg.num_lms,preds_pre,PIP_to_5)
                land5 = np.array(land5).astype(np.float32)
                land5_list.append(land5)

        if get_5point:
            return land_list, land5_list
        else:
            return land_list



    def get_fromImage(self,img,bbox,input_size=256,scale_mul=1.1):
        # norm N crop
        img, tmp_add, tmp_mul = crop_pip(img,bbox,target_size=input_size,scale_mul=scale_mul)
        

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        # torch input
        img = torch.tensor(img)
        img = torch.unsqueeze(img,dim=0)
        img_batch = img.shape[0]

        # inference
        output = self.net(img)
        if self.model_type=='trt':
            for i in range(len(output)):
                output[i] = np.reshape(output[i].ravel(),(img_batch,-1,8,8))

        # post process
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip_np(output,self.cfg.input_size, self.cfg.net_stride, self.cfg.num_nb)
        lms_pred = np.concatenate((lms_pred_x, lms_pred_y),axis=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
        tmp_x = np.mean(np.concatenate((lms_pred_x, tmp_nb_x), axis=1), axis=1).reshape(-1,1)
        tmp_y = np.mean(np.concatenate((lms_pred_y, tmp_nb_y), axis=1), axis=1).reshape(-1,1)
        lms_pred_merge = np.concatenate((tmp_x, tmp_y), axis=1).flatten()

        preds_pre=[]
        for j in range(self.cfg.num_lms):
            x_pred = lms_pred_merge[j*2]*tmp_mul[0] + tmp_add[0] # resize 인 경우 원래 이미지 크기 곱
            y_pred = lms_pred_merge[j*2+1]*tmp_mul[1] + tmp_add[1] # crop 돼서 포인트 기준이 바뀐 경우 x,y 바뀐만큼 더하기
            
            preds_pre.append([x_pred,y_pred])

        preds_pre = np.array(preds_pre).astype(np.float32)

        land5 = pip_get_5point(self.cfg.num_lms,preds_pre,PIP_to_5)
        land5 = np.array(land5).astype(np.float32)

        return preds_pre, land5

    def get_fromImage_notcrop(self,img,input_size=256):

        ori_size=img.shape
        ori_h = ori_size[0]
        ori_w = ori_size[1]
        

        # resize image
        img = cv2.resize(img,(input_size,input_size))

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        # torch input
        img = torch.tensor(img)
        img = torch.unsqueeze(img,dim=0)
        img_batch=img.shape[0]

        # inference
        output = self.net(img)
        if self.model_type=='trt':
            for i in range(len(output)):
                output[i] = np.reshape(output[i].ravel(),(img_batch,-1,8,8))

        # post process
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip_np(output,self.cfg.input_size, self.cfg.net_stride, self.cfg.num_nb)
        lms_pred = np.concatenate((lms_pred_x, lms_pred_y),axis=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].reshape(self.cfg.num_lms, self.max_len)
        tmp_x = np.mean(np.concatenate((lms_pred_x, tmp_nb_x), axis=1), axis=1).reshape(-1,1)
        tmp_y = np.mean(np.concatenate((lms_pred_y, tmp_nb_y), axis=1), axis=1).reshape(-1,1)
        lms_pred_merge = np.concatenate((tmp_x, tmp_y), axis=1).flatten()

        preds_pre=[]
        for j in range(self.cfg.num_lms):
            x_pred = lms_pred_merge[j*2]*ori_w # resize 인 경우 원래 이미지 크기 곱
            y_pred = lms_pred_merge[j*2+1]*ori_h # crop 돼서 포인트 기준이 바뀐 경우 x,y 바뀐만큼 더하기
            
            preds_pre.append([x_pred,y_pred])

        preds_pre = np.array(preds_pre).astype(np.float32)

        land5 = pip_get_5point(self.cfg.num_lms,preds_pre,PIP_to_5)
        land5 = np.array(land5).astype(np.float32)

        return preds_pre, land5

    