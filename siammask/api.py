from PIL import Image
import numpy as np
import os
from io import BytesIO
import base64
import re
import jsonpickle
import json
from .test import siamese_init, siamese_track
from .server import *
from .init import siammask, config

def encode_state(state):
    state['net.zf'] = state['net'].zf
    state.pop('net', None)
    state.pop('mask', None)

    for k,v in state.items():
        state[k] = jsonpickle.encode(v)

    return state

def decode_state(state):
    for k,v in state.items():
        state[k] = jsonpickle.decode(v)
    state['net'] = siammask
    state['net'].zf = state['net.zf']
    del state['net.zf']

    return state


def cvat_info():
    resp = {
            "framework":"pytorch",
            "spec": None,
            "type": "tracker",
            "description": "Fast Online Object Tracking and Segmentation"
    }
    return resp

def cvat_invoke(post_json):
    input_img_b64 = post_json["image"]
    state = post_json["state"]
    shape = post_json["shape"]
    image_data = re.sub('^data:image/.+;base64,', '', input_img_b64)
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    image = np.array(image)
    
    if state is None: # init tracking
        xtl, ytl, xbr, ybr = shape
        target_pos = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])
        target_sz = np.array([xbr - xtl, ybr - ytl])
        state = siamese_init(image, target_pos, target_sz, siammask,
            config['hp'])
        state = encode_state(state)
    else: # track
        state = decode_state(state)
        state = siamese_track(state, image, mask_enable=True,
            refine_enable=True)
        shape = state['ploygon'].flatten().tolist()
        state = encode_state(state)
    
    return {"shape": shape, "state": state}
