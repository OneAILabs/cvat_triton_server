import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', nargs='+', required=False, default=None,
                    help='Specify names, shapes, and types of model inputs tensors as {name}:{shape}:{type}. For example, input0:3,608,608:fp32. ' +
                    'Type is bool|uint8|uint16|uint32|uint64|int8|int16|int32|int64|fp16|fp32|fp64|string.' + 
                    'Do not be required for TensorRT, Tensorflow SavedModel, and ONNX model formats.')
parser.add_argument('-o', '--outputs', nargs='+', required=False, default=None,
                    help='Specify names, shapes, and types of model inputs tensors as {name}:{shape}:{type}. For example, input0:3,608,608:fp32. ' +
                    'Type is bool|uint8|uint16|uint32|uint64|int8|int16|int32|int64|fp16|fp32|fp64|string.' +
                    'Do not be required for TensorRT, Tensorflow SavedModel, and ONNX model formats.')
parser.add_argument('-m', '--model-name', type=str, required=True, default=None,
                    help='Name of model.')
parser.add_argument('--max-batch-size', type=int, required=False, default=1,
                    help='Specify max batch size. Default is 1.' )
parser.add_argument('-f', '--model-framework', type=str, required=False, default=None,
                    help='Framework of model to deploy (tensorrt_plan|tensorflow_graphdef|tensorflow_savedmodel|caffe2_netdef|onnxruntime_onnx|pytorch_libtorch).')
parser.add_argument('-b', '--model-backend', type=str, required=False, default=None,
                    help='Backend of model to deploy.')
parser.add_argument('-d', '--inference-device', type=str, required=False, default='CPU',
                    help='Choose CPU or GPU as inference device. Default is CPU')

parser.add_argument('-w', '--warmup-file', type=str, required=False, default='',
                    help='Warm up file path')


def main():
    args = parser.parse_args()
    model_dir = os.path.join('/models', args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_config_path = os.path.join(model_dir, 'config.pbtxt')
    with open(model_config_path, 'w') as f:
        f.write('name: \"' + args.model_name + '\"\n')
        if args.model_framework is not None:
            f.write('platform: \"' + args.model_framework + '\"\n')
        if args.model_backend is not None:
            f.write('backend: \"' + args.model_backend + '\"\n')
        f.write('max_batch_size: ' + str(args.max_batch_size) + '\n')
        if args.inputs:
            inputs = args.inputs
            if len(inputs) > 0:
                f.write('input [\n')
                for i, input in enumerate(inputs):
                    name = input.split(':')[0]
                    dims = input.split(':')[1]
                    data_type = input.split(':')[2]
                    f.write('  {\n')
                    f.write('    name: \"' + name + '\"\n')
                    f.write('    dims: [ ' + dims + ' ]\n')
                    f.write('    data_type: TYPE_' + data_type.upper() + '\n')
                    if i == len(inputs) - 1:
                        f.write('  }\n')
                    else:
                        f.write('  },\n')
                f.write(']\n')
        if args.outputs:
            outputs = args.outputs
            if len(outputs) > 0:
                f.write('output [\n')
                for i, output in enumerate(outputs):
                    name = output.split(':')[0]
                    dims = output.split(':')[1]
                    data_type = output.split(':')[2]
                    f.write('  {\n')
                    f.write('    name: \"' + name + '\"\n')
                    f.write('    dims: [ ' + dims + ' ]\n')
                    f.write('    data_type: TYPE_' + data_type.upper() + '\n')
                    if i == len(outputs) - 1:
                        f.write('  }\n')
                    else:
                        f.write('  },\n')
                f.write(']\n')
        f.write('instance_group [\n')
        f.write('  {\n')
        if args.inference_device == 'GPU':
            f.write('    kind: KIND_GPU,\n')
        else:
            f.write('    kind: KIND_CPU,\n')
        f.write('    count: 1\n')
        f.write('  }\n')
        f.write(']\n')
        if os.environ['WARM_UP'] == "enable" and args.warmup_file:
            with open(args.warmup_file, "r") as w:
                f.write(w.read())


if __name__ == '__main__':
    main()
