import os
import os
import numpy as np
import cv2
import time
import torch

from ..model_common import load_onnx, load_tensorRT, load_openvino, load_tensorRT_multiple
from ...utils.utils_pose.pose_util import preprocessing, keypoints_from_heatmaps, draw_pose, FLIP_PAIRS,NUM_JOINTS
from ...data.image import read_image

# from model_zoo.model_common import load_onnx, load_tensorRT, load_openvino, load_tensorRT_multiple
# from utils.utils_pose.pose_util import preprocessing, keypoints_from_heatmaps, draw_pose, FLIP_PAIRS,NUM_JOINTS
# from data.image import read_image

class VitPose:
    def __init__(self,model_type,model_path,**kwargs):

        self.model_path = model_path
        self.model_type = model_type
        self.load_multi = kwargs.get("load_multi",False)

        self.img_size = kwargs.get("img_size",(256,192)) # h,w

        self.dataset = kwargs.get("dataset",'coco')
        self.use_udp = kwargs.get("use_udp",True)

        self.num_joints = NUM_JOINTS[self.dataset]
        self.flip_pairs = FLIP_PAIRS[self.dataset]

        if model_type in ['vino','openvino']:
            self.model_type = 'openvino'
            self.model_name = model_path[0].split("/")[-1]
        else:
            self.model_name = model_path.split("/")[-1]

        # model load
        if self.model_type=='onnx':
            onnx_device=kwargs.get("onnx_device",'cuda')
            output_sort=kwargs.get("output_sort",True)
            self.net = load_onnx.Onnx(self.model_path,output_sort=output_sort,onnx_device=onnx_device)
        elif self.model_type=='trt':
            if self.load_multi:
                torch.cuda.initialized = True
                torch.cuda.is_available()
                torch.cuda.set_device(0)
                self.net = load_tensorRT_multiple.load(self.model_path)
            else:
                self.net = load_tensorRT.TrtModel(self.model_path)
        elif self.model_type=='openvino':
            if self.load_multi:
                self.net = load_openvino.Openvino_multi(self.model_path)
            else:
                vino_device=kwargs.get('vino_device','CPU')
                self.net = load_openvino.Openvino(self.model_path,device=vino_device)

    # input type
    # numpy array  / norm / (N,C,H,W) - onnx, vino-multiple, tensorRT
    # torch tensor / norm / (N,C,H,W) - vino
    # torch tensor / norm / (N,C,H,W) / input.cuda() - tensorRT-multiple
    def infer(self,img,bboxes,scores,toRGB=False,box_format='xyxy'): # input RGB image (original)

        if type(img)==str:
            img = read_image(img)
        if toRGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        input_datas = preprocessing(img.copy(),bboxes,scores,self.flip_pairs,image_size=[self.img_size[1],self.img_size[0]],num_joints=self.num_joints,use_udp=self.use_udp,box_format=box_format)
        centers = np.array([data['center'] for data in input_datas])
        scales = np.array([data['scale'] for data in input_datas])

        outs=[]
        for data in input_datas:
            img = data['img']
            out = self.net(img)[0]
            outs.append(np.squeeze(out,0))
        outs = np.array(outs)
        if self.model_type=='trt':
            outs = np.reshape(outs,(len(centers),self.num_joints,self.img_size[0]//4,self.img_size[1]//4))

        results = keypoints_from_heatmaps(outs,centers,scales,use_udp=self.use_udp)

        return results

    def infer_multi(self,img,bboxes,scores,toRGB=False,box_format='xyxy'):

        if type(img)==str:
            img = read_image(img)
        if toRGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        input_datas = preprocessing(img.copy(),bboxes,scores,self.flip_pairs,image_size=[self.img_size[1],self.img_size[0]],num_joints=self.num_joints,use_udp=self.use_udp,box_format=box_format)

        imgs = np.array([np.squeeze(data['img'],0) for data in input_datas])
        centers = np.array([data['center'] for data in input_datas])
        scales = np.array([data['scale'] for data in input_datas])

        outs = self.net(imgs)[0]
        outs = np.array(outs)
        if self.model_type=='trt':
            outs = np.reshape(outs,(img.shape[0],self.num_joints,self.img_size[0]//4,self.img_size[1]//4))

        results = keypoints_from_heatmaps(outs,centers,scales,use_udp=self.use_udp)

        return results


    def infer_benchmark(self,img,bboxes,scores,toRGB=False,box_format='xyxy'): # input RGB image (original)

        if type(img)==str:
            img = read_image(img)
        if toRGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        start_preproc=time.time()
        input_datas = preprocessing(img.copy(),bboxes,scores,self.flip_pairs,image_size=[self.img_size[1],self.img_size[0]],num_joints=self.num_joints,use_udp=self.use_udp,box_format=box_format)
        centers = np.array([data['center'] for data in input_datas])
        scales = np.array([data['scale'] for data in input_datas])
        end_preproc=time.time()
        

        start_forward=time.time()
        outs=[]
        for data in input_datas:
            img = data['img']
            out = self.net(img)[0]
            outs.append(np.squeeze(out,0))
        outs = np.array(outs)
        if self.model_type=='trt':
            out = np.reshape(out,(len(centers),self.num_joints,self.img_size[0]//4,self.img_size[1]//4))
        end_forward=time.time()

        start_decode=time.time()
        results = keypoints_from_heatmaps(outs,centers,scales,use_udp=self.use_udp)
        end_decode=time.time()

        time_preproc = (end_preproc-start_preproc)*1000
        time_forward = (end_forward-start_forward)*1000
        time_decode = (end_decode-start_decode)*1000

        return results, time_preproc,time_forward,time_decode

    def infer_multi_benchmark(self,img,bboxes,scores,toRGB=False,box_format='xyxy'): # input RGB image (original)

        if type(img)==str:
            img = read_image(img)
        if toRGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        start_preproc=time.time()
        input_datas = preprocessing(img.copy(),bboxes,scores,self.flip_pairs,image_size=[self.img_size[1],self.img_size[0]],num_joints=self.num_joints,use_udp=self.use_udp,box_format=box_format)
        imgs = np.array([np.squeeze(data['img'],0) for data in input_datas])
        centers = np.array([data['center'] for data in input_datas])
        scales = np.array([data['scale'] for data in input_datas])
        end_preproc=time.time()
        

        start_forward=time.time()
        outs = self.net(imgs)[0]
        outs = np.array(outs)
        if self.model_type=='trt':
            outs = np.reshape(outs,(img.shape[0],self.num_joints,self.img_size[0]//4,self.img_size[1]//4))
        end_forward=time.time()

        start_decode=time.time()
        results = keypoints_from_heatmaps(outs,centers,scales,use_udp=self.use_udp)
        end_decode=time.time()

        time_preproc = (end_preproc-start_preproc)*1000
        time_forward = (end_forward-start_forward)*1000
        time_decode = (end_decode-start_decode)*1000

        return results, time_preproc,time_forward,time_decode

        

