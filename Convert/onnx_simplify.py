import argparse
from onnx import checker
import onnx
from onnxsim import simplify

def convert2simplify(input_path, save_path,input_shape=(1,3,640,640)):

    model = onnx.load(input_path)
    graph = model.graph
    input_name = graph.input[0].name

    model, check = simplify(model, input_shapes={input_name:input_shape},dynamic_input_shape=False)

    onnx.save(model,save_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='onnx simplify')
    parser.add_argument('input', type=str, help='input model.onnx file')
    parser.add_argument('--output', type=str, default=None, help='output simplify onnx path')
    args = parser.parse_args()

    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.onnx")
    assert os.path.exists(input_file)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model_simplify.onnx")
    convert2simplify(input_file, args.output)
