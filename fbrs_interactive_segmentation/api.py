from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from io import BytesIO
import base64
import re
from .server import *

from .isegm.inference.predictors import get_predictor
from .isegm.inference.clicker import Clicker, Click
from .init import model, device

def convert_mask_to_polygon(mask):
    mask = np.array(mask, dtype=np.uint8)
    cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon


def cvat_info():
    resp = {
            "framework":"pytorch",
            "spec": None,
            "type": "interactor",
            "description": "f-BRS interactive segmentation"
    }
    return resp

def cvat_invoke(post_json):
    input_img_b64 = post_json["image"]
    pos_points = post_json["pos_points"]
    neg_points = post_json["neg_points"]
    threshold = post_json.get("threshold", 0.5)
    image_data = re.sub('^data:image/.+;base64,', '', input_img_b64)
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    image_nd = input_transform(image).to(device)
    clicker = Clicker()
    for x, y in pos_points:
        click = Click(is_positive=True, coords=(y, x))
        clicker.add_click(click)

    for x, y in neg_points:
        click = Click(is_positive=False, coords=(y, x))
        clicker.add_click(click)

    predictor_params = {
    'brs_mode': 'f-BRS-B',
    'brs_opt_func_params': {'min_iou_diff': 0.001},
    'lbfgs_params': {'maxfun': 20},
    'predictor_params': {'max_size': 800, 'net_clicks_limit': 8},
    'prob_thresh': threshold,
    'zoom_in_params': {'expansion_ratio': 1.4, 'skip_clicks': 1, 'target_size': 480}}
    
    predictor = get_predictor(model, device=device,
        **predictor_params)
    
    predictor.set_input_image(image_nd)

    object_prob = predictor.get_prediction(clicker)
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    object_mask = object_prob > threshold
    polygon = convert_mask_to_polygon(object_mask)
    #print(polygon)
    return polygon 
