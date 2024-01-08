import os
import argparse
import cv2
import numpy as np
import math
import torch

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.model_lib.MultiFTNet import MultiFTNet
from src.utility import get_kernel

import onnx

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="convert model path")
    parser.add_argument("--output_path",type=str, default="",help="onnx save path")
    parser.add_argument("--model_type",type=str,default='MultiFTNet')
    parser.add_argument("--num_classes",type=int,default=2)
    parser.add_argument("--w_input",type=int,default=80)
    parser.add_argument("--h_input",type=int,default=80)
    parser.add_argument("--opset",type=int,default=11)
    parser.add_argument("--not_simplify",action='store_false',default=True)
    args = parser.parse_args()

    return args


MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE,
    'MultiFTNet': MultiFTNet
}

args = get_args()
model_path = args.model_path
output_path = args.output_path
model_type = args.model_type
num_classes = args.num_classes
w_input, h_input = args.w_input,args.h_input

opset = args.opset
not_simplify = args.not_simplify

kernel_size = get_kernel(h_input, w_input,)
model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size,num_classes=num_classes)

# dummy input
img = np.random.randint(0, 255, size=(h_input, w_input, 3), dtype=np.uint8)
img = img.astype(np.float)
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0).float()

# model load
state_dict = torch.load(model_path)

keys = iter(state_dict)
first_layer_name = keys.__next__()

if first_layer_name.find('module.') >= 0:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name_key = key[7:]
        new_state_dict[name_key] = value
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)
model.eval()

# torch 2 onnx
torch.onnx.export(model, img, output_path, keep_initializers_as_inputs=True, verbose=False, opset_version=opset)

# simplify
if not not_simplify:
    print("Simplify run..")
    model = onnx.load(output_path)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

    from onnxsim import simplify
        
    input_shapes = {model.graph.input[0].name : (1, 3, h_input, w_input)}
    model, check = simplify(model, input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output_path)
