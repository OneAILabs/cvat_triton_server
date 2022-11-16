import os
import base64
from yolo.api import cvat_info, cvat_invoke
from yolo.data_processing import ALL_CATEGORIES

cvat_info_answer = {"framework": "yolov3" if os.environ['YOLO_VER'] == "V3" else "yolov4", "spec": ALL_CATEGORIES, "type": "detector", "description": "Object detetion via Yolov3" if os.environ['YOLO_VER'] == "V3" else "Object detetion via Yolov4"}

cvat_invoke_answer = [{'label': 'dog', 'points': [135, 220, 319, 544], 'type': 'rectangle', 'attributes': []}, {'label': 'bicycle', 'points': [80, 125, 616, 446], 'type': 'rectangle', 'attributes': []}, {'label': 'truck', 'points': [478, 81, 689, 168], 'type': 'rectangle', 'attributes': []}] if os.environ['YOLO_VER'] == "V3" else [{'label': 'dog', 'points': [127, 224, 313, 541], 'type': 'rectangle', 'attributes': []}, {'label': 'bicycle', 'points': [114, 128, 571, 427], 'type': 'rectangle', 'attributes': []}, {'label': 'pottedplant', 'points': [680, 109, 717, 154], 'type': 'rectangle', 'attributes': []}, {'label': 'truck', 'points': [464, 77, 686, 170], 'type': 'rectangle', 'attributes': []}]


def test_yolo():
    with open(os.path.join(os.environ['PROJECT_DIR'], "tests", "dog.jpg"), "rb") as f:
        data = f.read()
        data = base64.b64encode(data).decode()

    payload_dict = {"image": data}
    assert cvat_info() == cvat_info_answer
    results = cvat_invoke(payload_dict)
    print(results)
    assert results == cvat_invoke_answer
    

