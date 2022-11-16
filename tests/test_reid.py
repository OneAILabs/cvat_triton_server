import os
import base64
from reid.api import cvat_info, cvat_invoke

cvat_info_answer = {"framework":"openvino","spec": None,"type": "reid","description": "Person reidentification model for a general scenario"}
cvat_invoke_answer = [-1, 1, 0]

def test_reid():
    with open(os.path.join(os.environ['PROJECT_DIR'], "tests", "reid_test0.jpg"), "rb") as f1:
        data0 = f1.read()
        data0 = base64.b64encode(data0).decode()
    with open(os.path.join(os.environ['PROJECT_DIR'], "tests", "reid_test1.jpg"), "rb") as f2:
        data1 = f2.read()
        data1 = base64.b64encode(data1).decode()

    payload_dict = {"image0": data0, "image1": data1, "threshold": 0.5, "max_distance": 100, "boxes0": [{"points":[20,78,148,334], "label_id": 1}, {"points":[186,78,314,334], "label_id": 1}, {"points":[355,78,483,334], "label_id": 1}], "boxes1": [{"points":[270,80,398,336], "label_id": 1}, {"points":[112,80,240,336], "label_id": 1}]}
    assert cvat_info() == cvat_info_answer
    results = cvat_invoke(payload_dict)
    assert results == cvat_invoke_answer
    

