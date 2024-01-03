import time
import numpy as np
import tensorrt as trt
import onnx
import sys, os
import argparse

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()

def convert2trt(onnx_model_file, trt_model_file,not_printed=False):
    engine = build_engine(onnx_model_file, trt_model_file,not_printed=not_printed)

def build_engine(onnx_file_path, engine_file_path,not_printed=False):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, trt.Runtime(TRT_LOGGER) as runtime, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28 # 256MiB
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        if not not_printed:
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            if not not_printed:
                print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        if not not_printed:
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        #engine = builder.build_cuda_engine(network)
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='onnx2trt v8')
    parser.add_argument('input', type=str, help='input model.onnx file')
    parser.add_argument('--output', type=str, default=None, help='output trt path')
    args = parser.parse_args()

    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.onnx")
    assert os.path.exists(input_file)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.trt")
    convert2trt(input_file, args.output,not_printed=False)
