
import yaml
import pandas as pd
import os
from .server import detect



DATA_YAML = os.environ.get('DATA_YAML','default')
if(DATA_YAML == 'default'):
    print("DATA_YAML:{}".format(DATA_YAML))
    DATA_YAML = '/model/yolov7/data.yaml'

with open(DATA_YAML, 'r') as stream:
    try:
        loaded = yaml.load(stream,Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)
NAMES = loaded['names']
def cvat_info():
    specs = []
    index = 0
    for name in NAMES:
        specs.append({"id":index, "name":name})
        index += 1
    resp = {"framework": "yolov7", "spec": specs, "type": "detector", "description": "Object detetion via Yolov7"}
    return resp



def cvat_invoke(post_json):
    base64_img = post_json["image"]
    datas = pd.DataFrame(detect(base64_img,'yolov7','1'))
    results = []
    for index,row in datas.iterrows():
        results.append({"label":NAMES[row['label']], "points": [row['xmin'], row['ymin'], row['xmax'], row['ymax']], "type": "rectangle", "attributes": []})
    print("cvat_invoke:{}".format(results))

    return results




