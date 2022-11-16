import os
import imp
import json
from .isegm.inference.utils import load_deeplab_is_model_v2, load_hrnet_is_model_v2

if os.environ.get("TRITON_SERVER_CPU_ONLY") and os.environ["TRITON_SERVER_CPU_ONLY"] == "1":
    device = "cpu"
else:
    device = "cuda"

backbone = "auto"
try:
    json_dir = imp.find_module("cvat_custom")[1]
except:
    json_dir = os.path.join(os.environ['PROJECT_DIR'], "fbrs_interactive_segmentation")
json_path = os.path.join(json_dir, "resnet.json")
if os.path.exists(json_path):
    data = open(json_path, "rb").read()
    if data is not None:
        data = json.loads(data)
        model = load_deeplab_is_model_v2(device, data["backbone"], data["deeplab_ch"], data["aspp_dropout"])

elif os.path.exists(os.path.join(json_dir, "hrnet.json")):
    data = open(os.path.join(json_dir, "hrnet.json"), "rb").read()
    if data is not None:
        data = json.loads(data)
        model = load_hrnet_is_model_v2(device, backbone, data["width"], data["ocr_width"], data["small"])

