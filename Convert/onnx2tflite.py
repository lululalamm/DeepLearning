import os
import argparse

import onnx
from onnx_tf.backend import prepare

import tensorflow as tf



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openvino quantization')
    parser.add_argument('--onnx_path', type=str,default="sample.onnx", help='onnx path')
    parser.add_argument('--pb_path', type=str, default='sample.pb', help='pb path')
    parser.add_argument('--output',type=str,default='sample.tflite',help='output tflite path')
    
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_path)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(args.output)

    # Convert
    converter = tf.lite.TFLiteConverter.from_saved_model(args.output)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()

    # Save
    with open(args.output, 'wb') as f:
        f.write(tflite_model)

