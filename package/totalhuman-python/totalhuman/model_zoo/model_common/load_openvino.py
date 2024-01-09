from openvino.inference_engine import IECore
from openvino import runtime as ov
import os
import numpy as np



class Openvino:
    def __init__(self,model_path,**kwargs):
        self.xml_path = model_path[0]
        self.bin_path = model_path[1]
        self.device = kwargs.get("device",'CPU')

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
        # input (img) = torch.tensor / normalization / shape : (N,C,H,W)
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

        self.output_sort = kwargs.get('output_sort',False)
        
        self.core = ov.Core()
        self.net = self.core.read_model(self.xml_path)
        self.reshape = kwargs.get("reshape",None)

        if not self.reshape is None:
            self.net.reshape(self.reshape)
        
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
        
        # input (img) = np.array / normalization / shape : (N,C,H,W)
        # arcface img = ((img / 255) - 0.5) / 0.5
            
        input_tensor = ov.Tensor(img)
        _ = self.infer_request.infer([input_tensor])
        
        output_tensors = []
        for i in self.sort_idx:
            output_tensors.append(self.infer_request.get_output_tensor(i))
        
        output = [ np.array(o.data) for o in output_tensors]
        
        return output
