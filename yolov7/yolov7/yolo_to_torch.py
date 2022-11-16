import os
import sys
import shutil
import subprocess
from os import environ
from os.path import join
import torch


def copy_file_to_path(src,dst):
    shutil.copyfile(src, dst)

def copy_weights_to_temp(src,dst):
    print("source:{}\n,dst:{}\n".format(src,dst))
    copy_file_to_path(src,dst)


def main():
    weights_last_file_name = "last.pt"
    weights_best_file_name = "best.pt"
    user_weight = os.environ.get("WEIGHT","default")

    conf_thres = os.environ.get('CONF_THRES','0.35')
    iou_thres = os.environ.get('IOU-THRES','0.65')
    width = os.environ.get('WIDTH','640')
    height = os.environ.get('HEIGHT','640')
    temp_dir = os.environ.get('TEMP_DIR','/temp/')
    model_name = 'yolov7'

    weights_last_src_file_path =  os.path.join('/model/yolov7/weights',weights_last_file_name)
    weights_best_src_file_path =  os.path.join('/model/yolov7/weights',weights_best_file_name)
    weights_temp_file_path = os.path.join(temp_dir,'yolov7.pt')
    if(user_weight != "default"):
        weights_src_file_path = user_weight
    else:
        if(os.path.exists(weights_best_src_file_path)):
            weights_src_file_path = weights_best_src_file_path
        else:
            weights_src_file_path = weights_last_src_file_path
    
    model_dir = os.path.join('/models', model_name, '1')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    output_file_path = os.path.join(model_dir, 'model.pt')

    ##Copy weights to temp
    copy_weights_to_temp(weights_src_file_path,weights_temp_file_path)
    device_count = torch.cuda.device_count()
    device_string = ""
    if(device_count == 0):
        device_string = 'cpu'
    else:
        for i in range(0,int(device_count)):
            if(i+1 == device_count):
                device_string = device_string + str(i)
            else:
                device_string = device_string + str(i)+","
    ##Export model to TorchScript:
    cmd = [\
        "python export_torch.py " + " " + \
        "--device " + device_string+ " " + \
        "--weights " + weights_temp_file_path + " " + \
        "--grid " + \
        "--topk-all 100 " + \
        "--iou-thres "  +  iou_thres + " " + \
        "--conf-thres " + conf_thres + " " + \
        "--img-size " + height + " " + width + " " + \
        "--torch-out "  +output_file_path   
    ]
    print("cmd:{}".format(cmd))
    retcode = subprocess.call(cmd, shell=True)
    if (retcode == 0):
        print('Done.')


if __name__ == '__main__':
    main()