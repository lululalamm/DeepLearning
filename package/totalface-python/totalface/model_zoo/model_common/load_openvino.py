from openvino.inference_engine import IECore
from openvino import runtime as ov
import os
import numpy as np

from ...data.image import read_torchImage


class Openvino:
    def __init__(self,model_path,**kwargs):
        self.xml_path = model_path[0]
        self.bin_path = model_path[1]
        self.device = kwargs.get("device",'CPU')
        self.not_norm = kwargs.get("not_norm",False)
        self.torch_image = kwargs.get("torch_image",False)

        self.outputs_order = kwargs.get("outputs_order",None)

        self.ie = IECore()
        self.net = self.ie.read_network(self.xml_path,self.bin_path)

        self.outputs_name = self.net.outputs

        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        
        self.net.input_info[self.input_blob].precision = 'FP32'
        self.net.outputs[self.out_blob].precision = 'FP32'

        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)

        self.outs_len = len(self.net.outputs)

    def __call__(self,img):
        not_norm = self.not_norm
        if not self.torch_image:
            img = read_torchImage(img,not_norm=not_norm)
        outs = self.exec_net.infer(inputs={self.input_blob: img})

        if self.outputs_order:
            new_outs=[]
            for name in self.outputs_order:
                new_outs.append(outs[name])
        else:
            new_outs = list(outs.values())

        return new_outs

class Openvino_multi:
    def __init__(self,model_path,reshape=False,**kwargs):
        self.xml_path = model_path[0]

        self.not_norm = kwargs.get("not_norm",True)
        self.transform = kwargs.get("transform",True)
        self.torch_image = kwargs.get("torch_image",False)


        self.output_sort = kwargs.get('output_sort',False)
        
        self.core = ov.Core()
        self.net = self.core.read_model(self.xml_path)
        if reshape:
            self.input_shape = kwargs.get("input_shape",[1,3,-1,-1])
            self.net.reshape(self.input_shape)
        
        # for infer
        self.compiled_model = self.core.compile_model(model=self.net)
        self.infer_request = self.compiled_model.create_infer_request()
        
        # output
        self.output_list = self.compiled_model.outputs
        output_names_ori = [list(o.get_names())[0] for o in self.output_list]
        
        if self.output_sort:
            self.sort_idx = np.argsort(output_names_ori)
            self.output_names = [output_names_ori[i] for i in self.sort_idx]
        else:
            self.sort_idx = np.array(range(len(output_names_ori)))
            self.output_names = output_names_ori
            
    def __call__(self,img):
        
        if not self.not_norm:
            img = ((img / 255) - 0.5) / 0.5
        if self.torch_image:
            img = np.array(img)
        if self.transform:
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img,axis=0)
            
        input_tensor = ov.Tensor(img)
        _ = self.infer_request.infer([input_tensor])
        
        
        output_tensors = []
        for i in self.sort_idx:
            output_tensors.append(self.infer_request.get_output_tensor(i))
        
        
        output = [ np.array(o.data) for o in output_tensors]
        
        return output
