import onnx
import onnxruntime

import cv2
import os
import numpy as np

class Onnx:
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

        if self.output_sort:
            self.output_names = sorted(self.output_names_)
        else:
            self.output_names = self.output_names_


    def __call__(self,img):
        
        # input (img) = np.array / normalization / shape: (N,C,H,W)
        inp_dct = {self.input_name:img}
        outs = self.net.run(self.output_names, input_feed=inp_dct)

        return outs


