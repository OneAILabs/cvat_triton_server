import os
import yolo_to_onnx as yolo

parser = yolo.DarkNetParser()
layer_configs = parser.parse_cfg_file(os.environ['CONFIG_FILE_PATH'])
for name in layer_configs.keys():
    if 'yolo' in name:
        print(layer_configs[name]['anchors'])
        exit(0)
print('')
