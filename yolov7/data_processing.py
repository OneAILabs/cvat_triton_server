from pathlib import Path
import pandas as pd
import sys
import numpy as np
import cv2
from numpy import random
import base64
import torch
from utils.general import non_max_suppression,scale_coords,xyxy2xywh
from utils.plots import plot_one_box


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

def img2cv(img):
    data = base64.b64decode(img)
    data = np.fromstring(data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def postprocess(pred,raw_img,reshape_img,conf_thres=0.25,iou_thres=0.45):
    pred_torch = torch.from_numpy(pred)
    pred = non_max_suppression(pred_torch, conf_thres,iou_thres, classes=None, agnostic=False)
    im0 = raw_img
    img = reshape_img      
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    detected_objects = pd.DataFrame({})
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            for *xyxy, conf, cls in reversed(det):
                # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                detected_objects = detected_objects.append({
                    "xmin": int(xyxy[0]),
                    "ymin": int(xyxy[1]),
                    "xmax": int(xyxy[2]),
                    "ymax": int(xyxy[3]),
                    "conf": conf,
                    "label": int(cls)
                },ignore_index=True)
    print(detected_objects)
    return detected_objects

