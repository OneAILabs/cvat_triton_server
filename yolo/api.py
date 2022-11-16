import numpy as np
import os
from .server import detect
from .data_processing import PostprocessYOLO, ALL_CATEGORIES
from AIMakerMonitor import counter_inc, api_count_inc
def get_anchors():
    if os.environ['YOLO_VER'] == 'V4':
        default_anchors = [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)]
    else:
        default_anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)]
    anchors_env = os.environ.get('YOLO_ANCHORS')
    if anchors_env is not None:
        anchors = []
        tmp = anchors_env.split(',')
        for i in range(len(tmp)):
            if i % 2 == 0:
                anchors.append((float(tmp[i]), float(tmp[i+1])))
        if len(anchors) != 9:
            return default_anchors
        else:
            return anchors
    else:
        return default_anchors

def cvat_info():
    specs = []
    index = 0
    for category in ALL_CATEGORIES:
        specs.append({"id":index,"name":category})
        index = index+1

    if os.environ['YOLO_VER'] == 'V3':
        resp = {"framework": "yolov3", "spec": specs, "type": "detector", "description": "Object detetion via Yolov3"}
    elif os.environ['YOLO_VER'] == 'V4':
        resp = {"framework": "yolov4", "spec": specs, "type": "detector", "description": "Object detetion via Yolov4"}
    return resp


def cvat_invoke(post_json):
    api_count_inc()
    if os.environ['YOLO_VER'] == 'V4':
        model_name = "yolov4"
    else:
        model_name = "yolov3"
    outputs, image_raw, input_resolution_yolov3_HW = detect(post_json["image"], model_name)
    shape_orig_WH = image_raw.size

    anchors = get_anchors()
    if os.environ['YOLO_VER'] == 'V4':
        yolo_masks = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    else:
        yolo_masks = [(6, 7, 8), (3, 4, 5), (0, 1, 2)]
    if os.environ.get('CONFIDENCE_THRESH'):
        obj_threshold = float(os.environ['CONFIDENCE_THRESH'])
    else:
        obj_threshold = 0.3
    if os.environ.get('NMS_THRESH'):
        nms_threshold = float(os.environ['NMS_THRESH'])
    else:
        nms_threshold = 0.5
    postprocessor_args = {"yolo_masks": yolo_masks,
                          "yolo_anchors": anchors,
                          "obj_threshold": obj_threshold,
                          "nms_threshold": nms_threshold,
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)
    boxes, classes, scores = postprocessor.process(outputs, (shape_orig_WH))
    results = []
    if boxes is not None and len(boxes) > 0:
        for box, score, category in zip(boxes, scores, classes):
            x_coord, y_coord, width, height = box
            left = int(max(0, np.floor(x_coord + 0.5).astype(int)))
            top = int(max(0, np.floor(y_coord + 0.5).astype(int)))
            right = int(min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int)))
            bottom = int(min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int)))
            results.append({"label": ALL_CATEGORIES[category], "points": [left, top, right, bottom], "type": "rectangle", "attributes": []})
            counter_inc("object_detect", ALL_CATEGORIES[category])
    return results
