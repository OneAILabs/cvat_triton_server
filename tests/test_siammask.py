import os
import base64
from siammask.api import cvat_info, cvat_invoke

cvat_info_answer = {'framework': 'pytorch', 'spec': None, 'type': 'tracker', 'description': 'Fast Online Object Tracking and Segmentation'}
cvat_invoke_answer = [[311.54278564453124, 362.7304382324219, 291.268798828125, 111.33316040039062, 444.4820556640625, 98.97726440429688, 464.75604248046875, 350.3745422363281],[310.51641845703125, 363.6803283691406, 287.5491943359375, 111.04100036621094, 437.13116455078125, 97.442626953125, 460.098388671875, 350.08197021484375]]

def check_answer(shape, num):
    if num == 1:
        assert shape == cvat_invoke_answer[0]
    else:
        assert shape == cvat_invoke_answer[1]

def test_siammask():
    project_dir = os.environ['PROJECT_DIR']
    assert cvat_info() == cvat_info_answer
    with open(os.path.join(project_dir, "tests", "00000.jpg"), "rb") as f:
        data = f.read()
        data = base64.b64encode(data).decode()

    payload_dict = {"image": data, "state": None, "shape": [298, 110, 480, 366]}
    response = cvat_invoke(payload_dict)
    with open(os.path.join(project_dir, "tests", "00001.jpg"), "rb") as f1:
        data = f1.read()
        data = base64.b64encode(data).decode()
    payload_dict = {"image": data, "state": response['state'], "shape": response['shape']}
    response = cvat_invoke(payload_dict)
    check_answer(response['shape'], 1)
    with open(os.path.join(project_dir, "tests", "00002.jpg"), "rb") as f2:
        data = f2.read()
        data = base64.b64encode(data).decode()
    response = cvat_invoke(payload_dict)
    check_answer(response['shape'], 2)
