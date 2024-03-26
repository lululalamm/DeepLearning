import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn

from collections import OrderedDict
from backbones import get_cmt_model,get_agengeder_model
import argparse

def convert_onnx(net, path_module, output, opset=12, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    img = img.astype(np.float)
    img = np.transpose(img,(2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    weight = torch.load(path_module)
    
    if type(weight)==OrderedDict:
        try:
            net.load_state_dict(weight)
        except:
            new_state_dict = OrderedDict()
            for n, v in weight.items():
                name = n.replace("module.","") 
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(weight.module.state_dict())
    net.eval()
    torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        model, check = onnxsim.simplify(model, input_shapes={"input.1":(1, 3, 112, 112)})
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight',type=str, default='')
    parser.add_argument("--output",type=str, default='')
    args = parser.parse_args()

    weight = args.weight
    output = args.output


    if "cmt" in weight:
        backbone_onnx = get_cmt_model("",num_features=512)
    else:
        backbone_onnx = get_agengeder_model("",num_features=512)

    convert_onnx(backbone_onnx, weight, output, simplify=True)


