from PIL import Image
import numpy as np
import os
from io import BytesIO
import base64
import re
import math
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean, cosine
from .server import detect


def _match_boxes(box0, box1, distance):
    cx0 = (box0["points"][0] + box0["points"][2]) / 2
    cy0 = (box0["points"][1] + box0["points"][3]) / 2
    cx1 = (box1["points"][0] + box1["points"][2]) / 2
    cy1 = (box1["points"][1] + box1["points"][3]) / 2
    is_good_distance = euclidean([cx0, cy0], [cx1, cy1]) <= distance
    is_same_label = box0["label_id"] == box1["label_id"]

    return is_good_distance and is_same_label

def _match_crops(crop0, crop1):
    #embedding0 = model.infer(crop0)
    #embedding1 = model.infer(crop1)
    [embedding0] = detect([crop0], "reid")
    [embedding1] = detect([crop1], "reid")

    embedding0 = embedding0.reshape(embedding0.size)
    embedding1 = embedding1.reshape(embedding1.size)

    return cosine(embedding0, embedding1)

def _compute_similarity_matrix(image0, boxes0, image1, boxes1,
    distance):
    def _int(number, upper):
        return math.floor(np.clip(number, 0, upper - 1))

    DISTANCE_INF = 1000.0

    matrix = np.full([len(boxes0), len(boxes1)], DISTANCE_INF, dtype=float)
    for row, box0 in enumerate(boxes0):
        w0, h0 = image0.size
        xtl0, xbr0, ytl0, ybr0 = (
            _int(box0["points"][0], w0), _int(box0["points"][2], w0),
            _int(box0["points"][1], h0), _int(box0["points"][3], h0)
        )

        for col, box1 in enumerate(boxes1):
            w1, h1 = image1.size
            xtl1, xbr1, ytl1, ybr1 = (
                _int(box1["points"][0], w1), _int(box1["points"][2], w1),
                _int(box1["points"][1], h1), _int(box1["points"][3], h1)
            )

            if not _match_boxes(box0, box1, distance):
                continue

            crop0 = image0.crop((xtl0, ytl0, xbr0, ybr0))
            crop1 = image1.crop((xtl1, ytl1, xbr1, ybr1))
            crop0 = np.transpose(np.array(crop0), (2, 0, 1))
            crop1 = np.transpose(np.array(crop1), (2, 0, 1))
            crop0 = np.expand_dims(crop0, axis=0)
            crop1 = np.expand_dims(crop1, axis=0)
            matrix[row][col] = _match_crops(crop0, crop1)

    return matrix


def cvat_info():
    resp = {
            "framework":"openvino",
            "spec": None,
            "type": "reid",
            "description": "Person reidentification model for a general scenario"
    }
    return resp

def cvat_invoke(post_json):
    buf0 = BytesIO(base64.b64decode(re.sub('^data:image/.+;base64,', '', post_json["image0"])))
    buf1 = BytesIO(base64.b64decode(re.sub('^data:image/.+;base64,', '', post_json["image1"])))
    image0 = Image.open(buf0).convert('RGB')
    image1 = Image.open(buf1).convert('RGB')

    threshold = float(post_json.get("threshold", 0.5))
    max_distance = float(post_json.get("max_distance", 50))

    boxes0 = post_json["boxes0"]
    boxes1 = post_json["boxes1"]
    
    similarity_matrix = _compute_similarity_matrix(image0,
        boxes0, image1, boxes1, max_distance)
    row_idx, col_idx = linear_sum_assignment(similarity_matrix)
    results = [-1] * len(boxes0)
    for idx0, idx1 in zip(row_idx, col_idx):
        if similarity_matrix[idx0, idx1] <= threshold:
            results[idx0] = int(idx1)

    return results
