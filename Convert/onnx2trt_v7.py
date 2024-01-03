import time
import numpy as np
import tensorrt as trt
import onnx
import sys, os
import argparse

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()

def convert2trt(onnx_model_file, trt_model_file):
    engine = build_engine(onnx_model_file, trt_model_file)

def build_engine(onnx_file_path, engine_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 256MiB
        builder.max_batch_size = 1
        builder.fp16_mode = True
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None


        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input model.onnx file')
    parser.add_argument('--output', type=str, default=None, help='output trt path')
    args = parser.parse_args()

    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.onnx")
    assert os.path.exists(input_file)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.trt")
    convert2trt(input_file, args.output)
