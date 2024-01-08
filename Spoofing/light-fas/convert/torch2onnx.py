import os
import argparse
import torch
from utils import build_model, load_checkpoint, read_py_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="")
    parser.add_argument("--model_path",default="")
    parser.add_argument('--resize',type=int,default=128)
    parser.add_argument('--device',default='cuda')
    parser.add_argument("--save_path",default="save.onnx")
    parser.add_argument("--simplify",default=True,action='store_false')
    parser.add_argument("--scaling",default=1.0)

    args = parser.parse_args()
    return args



args = get_args()
config_path = args.config
model_path = args.model_path
save_path = args.save_path

resize = args.resize
img_size = (resize,resize)

device = args.device
if device=='cuda':
    device = 'cuda'
else:
    device = 'cpu'

# start
config = read_py_config(config_path)

dummy_input = torch.rand(size=(1,3,*img_size),device=device)

model = build_model(config, device, strict=True, mode='convert')
model.to(device)

# if config.data_parallel.use_parallel:
#     model = torch.nn.DataParallel(model, **config.data_parallel.parallel_params)
if config.data_parallel.use_parallel:
    use_parallel=True
else:
    use_parallel=False

load_checkpoint(model_path, model, map_location=torch.device(device),
                    optimizer=None, strict=True,use_parallel=use_parallel)

model.scaling = args.scaling
model.eval()


input_names = ["data"]
output_names = ["probs"]

torch.onnx.export(model, dummy_input, save_path, verbose=True,
                      input_names=input_names, output_names=output_names)

print("Successfully exported onnx:",save_path)

if args.simplify:
    from onnxsim import simplify

    model = onnx.load(save_path)
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model, save_path)
    print("Successfully exported onnx simplify:",save_path)


