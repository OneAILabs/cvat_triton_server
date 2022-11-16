import os
import argparse
import torch

from isegm.inference.utils import load_deeplab_is_model, load_hrnet_is_model

def transform_to_onnx(model, onnxfile, batch_size, input_names, output_names, inputs, dynamic_axes):

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          inputs[0] if len(inputs) == 1 else inputs,
                          onnxfile,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')

    else:
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          inputs[0] if len(inputs) == 1 else inputs,
                          onnxfile,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')

def main():
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if os.environ['INFERENCE_DEVICE'] == 'GPU 'else 'cpu'
    args = parse_args()

    checkpoint_path = args.checkpoint
    import time
    start_time = time.time()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print("--- %s seconds ---" % (time.time() - start_time))

    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    model = None
    backbone = 'auto'
    for k in state_dict.keys():
        if 'feature_extractor.stage2.0.branches' in k:
            model = load_hrnet_is_model(state_dict, device, backbone)
            input1 = torch.randn(2, 5, 480, 854)
            inputs = (input1)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "rgb_conv.onnx")
            transform_to_onnx(model.rgb_conv, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

            input1 = torch.randn(2, 3, 480, 854)
            inputs = (input1)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "feature_extractor.compute_hrnet_feats.onnx")
            transform_to_onnx(model.feature_extractor, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

            input1 = torch.randn(2, 270, 120, 214)
            inputs = (input1)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "feature_extractor.aux_head.onnx")
            transform_to_onnx(model.feature_extractor.aux_head, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

            input1 = torch.randn(2, 270, 120, 214)
            inputs = (input1)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "feature_extractor.conv3x3_ocr.onnx")
            transform_to_onnx(model.feature_extractor.conv3x3_ocr, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

            input1 = torch.randn(2, 128, 120, 214)
            input2 = torch.randn(2, 1, 120, 214)
            inputs = (input1, input2)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "input2": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "feature_extractor.ocr_gather_head.onnx")
            transform_to_onnx(model.feature_extractor.ocr_gather_head, onnx_path, 0, ['input1', 'input2'], ['output1'], inputs, dynamic_axes)

            input1 = torch.randn(2, 128, 120, 214)
            input2 = torch.randn(2, 128, 1, 1)
            inputs = (input1, input2)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "input2": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "feature_extractor.ocr_distri_head.onnx")
            transform_to_onnx(model.feature_extractor.ocr_distri_head, onnx_path, 0, ['input1', 'input2'], ['output1'], inputs, dynamic_axes)
 
            input1 = torch.randn(2, 128, 120, 214)
            inputs = (input1)
            dynamic_axes = {"input1": {0: "batch_size", 1: "num", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
            onnx_path = os.path.join(outdir, "feature_extractor.cls_head.onnx")
            transform_to_onnx(model.feature_extractor.cls_head, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

            break

    if model is None:
        model = load_deeplab_is_model(state_dict, device, backbone)

        input1 = torch.randn(2, 5, 480, 854)
        inputs = (input1)
        dynamic_axes = {"input1": {0: "ya", 1: "test", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
        onnx_path = os.path.join(outdir, "rgb_conv.onnx")
        transform_to_onnx(model.rgb_conv, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)
        
        input1 = torch.randn(2, 3, 480, 854)
        inputs = (input1)
        dynamic_axes = {"input1": {0: "ya", 1: "test", 2: "h", 3: "w"}, "output1": {0: "batch_size"}, "output2": {0: "batch_size"}, "output3": {0: "batch_size"}, "output4": {0: "batch_size"}}
        onnx_path = os.path.join(outdir, "feature_extractor.backbone.onnx")
        transform_to_onnx(model.feature_extractor.backbone, onnx_path, 0, ['input1'], ['output1', 'output2', 'output3', 'output4'], inputs, dynamic_axes)
        
        input1 = torch.randn(2, 64, 120, 214)
        inputs = (input1)
        dynamic_axes = {"input1": {0: "ya", 1: "test", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
        onnx_path = os.path.join(outdir, "feature_extractor.skip_project.onnx")
        transform_to_onnx(model.feature_extractor.skip_project, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

        input1 = torch.randn(2, 512, 60, 107)
        inputs = (input1)
        dynamic_axes = {"input1": {0: "ya", 1: "test", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
        onnx_path = os.path.join(outdir, "feature_extractor.aspp.onnx")
        transform_to_onnx(model.feature_extractor.aspp, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

        input1 = torch.randn(2, 160, 120, 214)
        inputs = (input1)
        dynamic_axes = {"input1": {0: "ya", 1: "test", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
        onnx_path = os.path.join(outdir, "feature_extractor.head.onnx")
        transform_to_onnx(model.feature_extractor.head, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

        input1 = torch.randn(2, 128, 120, 214)
        inputs = (input1)
        dynamic_axes = {"input1": {0: "ya", 1: "test", 2: "h", 3: "w"}, "output1": {0: "batch_size"}}
        onnx_path = os.path.join(outdir, "head.onnx")
        transform_to_onnx(model.head, onnx_path, 0, ['input1'], ['output1'], inputs, dynamic_axes)

        
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. ')

    parser.add_argument('--outdir', type=str, required=True,
                        help='The directory of output models. ')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

