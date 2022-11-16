import sys
import numpy as np
import os
from attrdict import AttrDict
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
from .data_processing import preprocess,img2cv,postprocess

url = "localhost:8001"
protocol = "grpc"

conf_thres = float(os.environ.get('CONF_THRES','0.35'))
iou_thres = float(os.environ.get('IOU-THRES','0.65'))
width = int(os.environ.get('WIDTH','640'))
height = int(os.environ.get('HEIGHT','640'))

INPUT_NAMES = ["INPUT__0"]
OUTPUT_NAMES = ["OUTPUT__0"]


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


def detect(img_in_b64, model_name, model_version='1'):
    
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
    inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, height, width], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
    raw_img = img2cv(img_in_b64)
    if raw_img is None:
        print(f"FAILED: could not load input image")
        sys.exit(1)
    reshape_img = preprocess(raw_img, [ height,width])
    reshape_img = np.expand_dims(reshape_img, axis=0)

    inputs[0].set_data_from_numpy(reshape_img)    

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

    out = postprocess(pred,raw_img,reshape_img,conf_thres,iou_thres)
    return out

    
    