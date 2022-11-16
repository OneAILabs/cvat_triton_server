import argparse
import time
from pathlib import Path
import sys
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

INPUT_NAMES = ["INPUT__0"]
OUTPUT_NAMES = ["OUTPUT__0"]
NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
       ]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

def preprocess(img, input_shape, letter_box=True):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def triton(path):
    try:
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001",
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, opt.img_size, opt.img_size], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))

    print("Creating buffer from image file...")
    input_image = cv2.imread(str(path))
    if input_image is None:
        print(f"FAILED: could not load input image {str(path)}")
        sys.exit(1)
    input_image_buffer = preprocess(input_image, [640, 640])
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

    inputs[0].set_data_from_numpy(input_image_buffer)

    print("Invoking inference...")
    results = triton_client.infer(model_name='yolov7',
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=None)
    statistics = triton_client.get_inference_statistics(model_name='yolov7')
    if len(statistics.model_stats) != 1:
        print("FAILED: get_inference_statistics")
        sys.exit(1)
    print(statistics)

    for output in OUTPUT_NAMES:
        result = results.as_numpy(output)
        print(f"Received result buffer \"{output}\" of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")
    pred = results.as_numpy(OUTPUT_NAMES[0])
    pred_torch = torch.from_numpy(pred)
    print("pred:{}".format(pred.shape))
    print("pred_torch:{}".format(pred_torch.shape))
    pred = non_max_suppression(pred_torch, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)
    # Process detections
    im0 = input_image
    img = input_image_buffer      
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            print("im0.shape:{}".format(im0.shape))
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            for *xyxy, conf, cls in reversed(det):
                # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format

                # Add bbox to image
                label = f'{NAMES[int(cls)]} {conf:.2f}'
                print("xyxy:{}".format(xyxy))
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Save results (image with detections)
            cv2.imwrite("./result.jpg", im0)
            print(f" The image with the result is saved in: ./result.jpg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images/horses.jpg', help='source')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    opt = parser.parse_args()
    print(opt)
    triton(opt.source)

