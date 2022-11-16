from PIL import Image
import numpy as np
import os
from io import BytesIO
import base64
import re
import json
from skimage.measure import find_contours, approximate_polygon
from .server import *
from .init import model
from .utils import load_class_names

labels = load_class_names(os.environ['LABEL_FILE_PATH'])

def cvat_info():
    specs = []
    index = 0
    for lable in labels:
        specs.append({"id":index,"name":lable})
        index = index+1
    resp = {
            "framework":"tensorflow",
            "spec": specs,
            "type": "detector",
            "description": "Mask RCNN"
    }
    return resp

def cvat_invoke(post_json):
    input_img_b64 = post_json["image"]
    threshold = float(post_json.get("threshold", 0.2))
    image_data = re.sub('^data:image/.+;base64,', '', input_img_b64)
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    image = np.array(image)

    #model = modellib.MaskRCNN(mode="inference",
    #        config=InferenceConfig(), model_dir=LOG_DIR)
    output = model.detect([image], verbose=1)[0]

    results = []
    MASK_THRESHOLD = 0.5
    for i in range(len(output["rois"])):
        score = output["scores"][i]
        class_id = output["class_ids"][i]
        mask = output["masks"][:, :, i]
        if score >= threshold:
            mask = mask.astype(np.uint8)
            contours = find_contours(mask, MASK_THRESHOLD)
            # only one contour exist in our case
            contour = contours[0]
            contour = np.flip(contour, axis=1)
            # Approximate the contour and reduce the number of points
            contour = approximate_polygon(contour, tolerance=2.5)
            if len(contour) < 6:
                continue
            label = labels[class_id]

            results.append({
                "confidence": str(score),
                "label": label,
                "points": contour.ravel().tolist(),
                "type": "polygon",
            })

    return results
