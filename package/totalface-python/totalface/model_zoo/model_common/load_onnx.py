import onnx
import onnxruntime

import cv2
import os
import numpy as np


class Onnx_cv:
    def __init__(self,model_path,**kwargs):
        self.model_file = model_path
        self.net = cv2.dnn.readNetFromONNX(self.model_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.output_sort=kwargs.get("output_sort",False)

        self.layer_names = self.net.getLayerNames()
        self.output_names_ = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        if self.output_sort:
            self.output_names = sorted(self.output_names_)
        else:
            self.output_names = self.output_names_
        self.outs_len = len(self.output_names)

        self.input_mean = kwargs.get("input_mean",127.5)
        self.input_std = kwargs.get("input_std",128.0)


    def __call__(self,img,**kwargs):
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=kwargs.get('to_bgr',False))

        self.net.setInput(blob)

        outs = self.net.forward(self.output_names)

        return outs



class Onnx_session:
    def __init__(self,model_path,**kwargs):
        self.device = kwargs.get("onnx_device","cuda")
        if self.device=='cuda':
            self.providers=['CUDAExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
        print("providers:",self.providers)
        self.net = onnxruntime.InferenceSession(model_path,providers=self.providers)
        self.input_name = self.net.get_inputs()[0].name
        self.output_names_= [ output.name for output in self.net.get_outputs() ]
        self.outs_len = len(self.output_names_)

        self.output_sort=kwargs.get("output_sort",False)

        self.input_mean = kwargs.get("input_mean",127.5)
        self.input_std = kwargs.get("input_std",128.0)

        if self.output_sort:
            self.output_names = sorted(self.output_names_)
        else:
            self.output_names = self.output_names_

        self.torch_image = kwargs.get("torch_image",False)


    def __call__(self,img):
        
        if self.torch_image: # type torch, transpose, expand (N,C,H,W)
            img = np.array(img,dtype=np.float32)
        
        else:
            img = img.astype(np.float32).transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0).astype(np.float32) #NCHW
            
        img = (img - self.input_mean) / self.input_std
        inp_dct = {self.input_name:img}
        outs = self.net.run(self.output_names, input_feed=inp_dct)

        return outs


