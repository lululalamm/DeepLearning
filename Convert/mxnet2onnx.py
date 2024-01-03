import argparse
import os

import numpy as np
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet

def mxnet2onnx(base,sym,params,output,input_shape=(1,3,640,640),dynamic=False):

    if dynamic:
        converted_model = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, \
                                              output,dynamic=True,dynamic_input_shapes=[(1,3,None,None)])
    else:
        converted_model = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, \
                                              output,dynamic=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='onnx simplify')
    parser.add_argument('--base', type=str, help='mxnet model base directory')
    parser.add_argument('--sym', type=str, default=None, help='mxnet symbole file name')
    parser.add_argument('--params', type=str, default=None, help='mxnet params file name')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    args = parser.parse_args()

    base = args.base
    sym = os.path.join(base,args.sym)
    params = os.path.join(args.params)
    output = args.output

    mxnet2onnx(base,sym,params,output)
