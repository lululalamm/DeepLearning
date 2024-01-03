import os
import torch
import numpy as np
import argparse

import tensorrt as trt

from utils import utils_to, utils_flatten, utils_get_names, load_model

import sys


def get_args():
    parser = argparse.ArgumentParser(description="Speed Check HRNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_paths",type=str,nargs='+',default="",help="convert model paths")
    args = parser.parse_args()
    return args

def onnx2trt(
        model,
        log_level='ERROR',
        max_batch_size=1,
        min_input_shapes=None,
        max_input_shapes=None,
        max_workspace_size=1,
        fp16_mode=False,
        strict_type_constraints=False,
        output_sort=True):
    """build TensorRT model from Onnx model.

    Args:
        model (string or io object): Onnx model name
        log_level (string, default is ERROR): TensorRT logger level, now
            INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
        max_batch_size (int, default=1): The maximum batch size which can be
            used at execution time, and also the batch size for which the
            ICudaEngine will be optimized.
        min_input_shapes (list, default is None): Minimum input shapes, should
            be provided when shape is dynamic. For example, [(3, 224, 224)] is
            for only one input.
        max_input_shapes (list, default is None): Maximum input shapes, should
            be provided when shape is dynamic. For example, [(3, 224, 224)] is
            for only one input.
        max_workspace_size (int, default is 1): The maximum GPU temporary
            memory which the ICudaEngine can use at execution time. default is
            1GB.
        fp16_mode (bool, default is False): Whether or not 16-bit kernels are
            permitted. During engine build fp16 kernels will also be tried when
            this mode is enabled.
        strict_type_constraints (bool, default is False): When strict type
            constraints is set, TensorRT will choose the type constraints that
            conforms to type constraints. If the flag is not enabled higher
            precision implementation may be chosen if it results in higher
            performance.
    """

    logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if isinstance(model, str):
        with open(model, 'rb') as f:
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(model.read())
    if not flag:
        for error in range(parser.num_errors):
            print(parser.get_error(error))

    # re-order output tensor
    if output_sort:
        output_tensors_ = [network.get_output(i)
                      for i in range(network.num_outputs)]

        output_names = [o.name for o in output_tensors_]
        output_sorted = sorted(range(len(output_names)),key=lambda k: output_names[k])

        output_tensors=[ network.get_output(i) for i in output_sorted]
    else:
        output_tensors = [network.get_output(i)
                          for i in range(network.num_outputs)]
    [network.unmark_output(tensor) for tensor in output_tensors]
    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = 'identity_{}'.format(tensor.name)
        network.mark_output(tensor=identity_out_tensor)

    builder.max_batch_size = max_batch_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size * (1 << 25)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    # set dynamic shape profile
    assert not (bool(min_input_shapes) ^ bool(max_input_shapes))

    profile = builder.create_optimization_profile()

    input_shapes = [network.get_input(i).shape[1:]
                    for i in range(network.num_inputs)]
    if not min_input_shapes:
        min_input_shapes = input_shapes
    if not max_input_shapes:
        max_input_shapes = input_shapes

    assert len(min_input_shapes) == len(max_input_shapes) == len(input_shapes)
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        min_shape = (1,) + min_input_shapes[i]
        #min_shape = min_input_shapes[i]
        max_shape = (max_batch_size,) + max_input_shapes[i]
        opt_shape = [(min_ + max_) // 2
                     for min_, max_ in zip(min_shape, max_shape)]
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)

    return engine,config

args = get_args()

src_list=args.model_paths

trt_version = trt.__version__[0]

for src_path in src_list:
    dst_path = src_path.split(".onnx")[0]+".v{}.trt"
    dst_path = dst_path.format(trt_version)

    engine,_ = onnx2trt(src_path,max_batch_size=1, \
    min_input_shapes=[(3, 128, 128)], max_input_shapes=[(3, 1664, 1664)], \
    fp16_mode=False)

    with open(dst_path, "wb") as f:
        f.write(engine.serialize())

    print("Save",dst_path)
    #print("TRT Version:",trt_version)

