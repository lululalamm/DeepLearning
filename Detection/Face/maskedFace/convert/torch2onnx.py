import os
import numpy as np
import torch
import onnx
import argparse

from backbone.get_models import get_model
from collections import OrderedDict

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def convert_onnx(net, path_module, output, opset=11, simplify=False,img_size=224):
    assert isinstance(net, torch.nn.Module)

    img = np.random.randint(0, 255, size=(img_size, img_size, 3), dtype=np.int32)
    img = normalization(img)
    img = torch.from_numpy(img).unsqueeze(0).float()
    print(img.shape)

    load_weight = torch.load(path_module)
    new_state_dict = OrderedDict()
    for n, v in load_weight.items():
        name = n.replace("module.","") 
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()

    torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        # model, check = simplify(model)
        input_shapes = {model.graph.input[0].name : (1, 3, img_size, img_size)}
        model, check = simplify(model, input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Torch 2 Onnx')
    parser.add_argument('--input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--model_name', type=str, default=None, help='backbone network')
    parser.add_argument('--load_type',type=int,default=1)
    parser.add_argument('--num_classes',type=int,default=2)
    parser.add_argument('--opset', type=int, default=11, help='opset version')
    parser.add_argument('--simplify', type=bool, default=True, help='onnx simplify')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    model_name = args.model_name

    if "mnasnet" in model_name:
        print("mnasnet break")
    
    if model_name=='mobilefacenet':
        img_size=112
    else:
        img_size=224
    

    # load model
    backbone_onnx = get_model(model_name,args.num_classes,load_type=args.load_type)
    convert_onnx(backbone_onnx, input_path, output_path, args.opset, simplify=args.simplify,img_size=img_size)