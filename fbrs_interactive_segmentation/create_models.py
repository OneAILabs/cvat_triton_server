import sys
import os
import pathlib

from shutil import copyfile

def main(args):
    flist = []
    for (dirpath, dirnames, filenames) in os.walk(args[1]):
        for filename in filenames:
            flist.append(os.path.join(args[1], filename))
        break

    for f in flist:
        dirpath = os.path.join("/models", os.path.basename(f).split(".onnx")[0], "1")
        pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(dirpath, "model.onnx")
        copyfile(f, filepath)

        cmd = "python3 ../build_config.py -m " + os.path.basename(f).split(".onnx")[0] + " --max-batch-size 8 -f onnxruntime_onnx -d " + os.environ["INFERENCE_DEVICE"] + " -w ../models_warmup/fbrs_interactive_segmentation/" + os.path.basename(f).split(".onnx")[0] + ".pbtxt"
        os.system(cmd)

if __name__ == '__main__':
    args = sys.argv
    main(args)

