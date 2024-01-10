import os
import numpy as np

import os
import cv2
import time
import torch

from ..model_common import load_onnx, load_tensorRT, load_openvino, load_tensorRT_multiple
from ...utils.utils_detection.yolo_util import letterbox, non_max_suppression_np, scale_coords_np
from ...data.image import read_image
# from model_zoo.model_common import load_onnx, load_tensorRT, load_openvino, load_tensorRT_multiple
# from utils.utils_detection.yolo_util import letterbox, non_max_suppression_np, scale_coords_np
# from data.image import read_image

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# same Yolov7
class Yolov5:
    def __init__(self,model_type,model_path,**kwargs):

        self.model_path = model_path
        self.model_type = model_type
        self.load_multi = kwargs.get("load_multi",False)

        self.img_size = kwargs.get("img_size",(640,640))
        self.stride = kwargs.get("stride",32)
        self.conf_thres = kwargs.get("conf_thres",0.55)
        self.iou_thres = kwargs.get("iou_thres",0.45)
        self.classes = kwargs.get("classes",80)
        self.agnostic_nms = kwargs.get("agnostic_nms",False)
        self.multi_label = kwargs.get("multi_label",True)

        self.category = COCO_CATEGORY

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
    def detect(self,image,person=False): # image - RGB image array or path

        if type(image)==str:
            image = read_image(image)
        raw_shape = image.shape
        image = letterbox(image,self.img_size,stride=self.stride)[0] # padding
        new_shape = image.shape
        image = np.ascontiguousarray(image)

        # norm
        image = image.astype(np.float32)
        image /= 255.0  # 0 - 255 to 0.0 - 1.0  
        # transpose
        image = image.transpose(2, 0, 1) # (C,H,W)
        image = np.expand_dims(image,axis=0) # (N,C,H,W)

        # input type 
        if self.model_type=='openvino': # vino
            image = torch.tensor(image)

        prediction = self.net(image)[0] # predict

        if self.model_type=='trt':
           prediction = np.reshape(prediction,(prediction.shape[0],-1,self.classes+5))
        
        result = non_max_suppression_np(prediction, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms,multi_label=self.multi_label)[0]

        if result is not None:
            bboxes = result[:,:4]
            scores = result[:,4]
            labels = result[:,-1]
            new_bboxes = scale_coords_np(new_shape,bboxes,raw_shape)

            if person:
                choice_indexs = np.where(labels==0)[0]
                new_bboxes = new_bboxes[choice_indexs]
                scores = scores[choice_indexs]
                labels = labels[choice_indexs]
        else:
            return np.array([]),np.array([]),np.array([])

        return new_bboxes,scores,labels

    def detect_moteval(self,pimage,raw_shape,person=False): # image - RGB image array or path

        if self.model_type=='openvino':
            pimage = pimage[0]
        else:
            pimage = pimage[0].numpy().astype(np.float32)

        new_shape = pimage.shape[-2:]
        prediction = self.net(pimage)[0] # predict

        if self.model_type=='trt':
           prediction = np.reshape(prediction,(prediction.shape[0],-1,self.classes+5))
        
        result = non_max_suppression_np(prediction, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms,multi_label=self.multi_label)[0]

        if result is not None:
            bboxes = result[:,:4]
            scores = result[:,4]
            labels = result[:,-1]
            new_bboxes = scale_coords_np(new_shape,bboxes,raw_shape)

            if person:
                choice_indexs = np.where(labels==0)[0]
                new_bboxes = new_bboxes[choice_indexs]
                scores = scores[choice_indexs]
                labels = labels[choice_indexs]
            return new_bboxes,scores,labels
        else:
            return np.array([]),np.array([]),np.array([])

    
    def detect_multiple(self,images): # image - RGB image array (N,H,W,C) or path list (N)

        if not self.load_multi:
            print("detect_multiple is 'load multi' must True")
            return

        raw_shapes=[]
        new_shapes=[]
        if type(images[0])==str:
            image_list = []
            for img in images:
                image_list.append(read_image(img))
            images = image_list

        input_list=[]
        for image in images:
            if len(image.shape)>3:
                image = np.squeeze(image,axis=0)
            raw_shapes.append(image.shape)
            image = letterbox(image,self.img_size,stride=self.stride)[0] # padding
            new_shapes.append(image.shape)
            image = np.ascontiguousarray(image)

            # norm
            image = image.astype(np.float32)
            image /= 255.0  # 0 - 255 to 0.0 - 1.0  
            # transpose
            image = image.transpose(2, 0, 1) # (C,H,W)
            if self.model_type=='trt':
                image = torch.tensor(image)
                image = image.cuda()
            input_list.append(image)
        
        input_list = np.array(input_list)
        prediction = self.net(input_list)[0] # predict # (N,-1,classes+5)

        if self.model_type=='trt':
            for i in range(len(prediction)):
                prediction[i] = np.reshape(prediction[i],(prediction[i].shape[0],-1,self.classes+5))
                prediction[i] = prediction[i].cpu().numpy()

        new_bboxes=[]
        scores=[]
        labels=[]
        for i in range(len(prediction)):
            pred = np.expand_dims(prediction[i],axis=0)
            result = non_max_suppression_np(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms,multi_label=self.multi_label)
            if result is not None:
                bbox = result[:,:4]
                score = result[:,4]
                label = result[:,-1]
                new_bbox = scale_coords_np(new_shapes[i],bbox,raw_shapes[i])

                if person:
                    choice_indexs = np.where(label==0)[0]
                    new_bbox = new_bbox[choice_indexs]
                    score = score[choice_indexs]
                    label = label[choice_indexs]
            else:
                new_bbox=[]
                score=[]
                label=[]
            new_bboxes.append(new_bbox)
            scores.append(score)
            labels.append(label)


        return np.array(new_bboxes),np.array(scores),np.array(labels)