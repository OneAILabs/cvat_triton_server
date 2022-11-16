import os
import base64
from fbrs_interactive_segmentation.api import cvat_info, cvat_invoke
from fbrs_interactive_segmentation.isegm.inference.clicker import Click

clicks = [Click(is_positive=True, coords=(201, 605)), Click(is_positive=True, coords=(320, 623)), Click(is_positive=False, coords=(323, 658)), Click(is_positive=True, coords=(341, 658)), Click(is_positive=True, coords=(296, 615)), Click(is_positive=True, coords=(170, 574)), Click(is_positive=True, coords=(200, 622)), Click(is_positive=True, coords=(281, 570)), Click(is_positive=False, coords=(289, 549)), Click(is_positive=False, coords=(290, 548)), Click(is_positive=False, coords=(289, 548)), Click(is_positive=False, coords=(290, 547)), Click(is_positive=False, coords=(290, 549)), Click(is_positive=False, coords=(288, 550)), Click(is_positive=False, coords=(291, 548)), Click(is_positive=False, coords=(288, 549)), Click(is_positive=False, coords=(289, 550)), Click(is_positive=False, coords=(291, 549)), Click(is_positive=False, coords=(289, 547)), Click(is_positive=False, coords=(290, 546))]

cvat_info_answer = {"framework":"pytorch","spec": None,"type": "interactor","description": "f-BRS interactive segmentation"}

cvat_invoke_answer = [[595, 150], [590, 151], [587, 152], [581, 155], [572, 161], [567, 166], [563, 171], [561, 174], [561, 176], [563, 178], [565, 179], [570, 180], [575, 180], [578, 183], [580, 188], [581, 191], [582, 195], [583, 208], [583, 211], [582, 215], [581, 218], [579, 223], [578, 225], [574, 229], [572, 232], [571, 236], [570, 238], [565, 243], [565, 245], [566, 247], [569, 248], [573, 250], [575, 252], [576, 254], [576, 258], [577, 259], [577, 262], [576, 266], [575, 268], [574, 270], [571, 273], [568, 275], [566, 276], [563, 277], [558, 278], [554, 277], [546, 274], [543, 274], [541, 275], [537, 278], [536, 279], [535, 281], [532, 286], [531, 288], [530, 293], [530, 305], [531, 310], [532, 313], [538, 321], [537, 325], [539, 328], [542, 331], [546, 334], [549, 335], [553, 335], [556, 336], [559, 338], [563, 339], [569, 339], [572, 338], [577, 334], [579, 332], [581, 329], [582, 327], [583, 325], [584, 318], [585, 315], [587, 313], [587, 302], [588, 294], [593, 289], [595, 289], [598, 292], [601, 293], [601, 295], [604, 300], [605, 303], [605, 305], [602, 309], [602, 313], [605, 319], [611, 325], [611, 329], [613, 331], [617, 332], [620, 333], [626, 333], [629, 336], [629, 341], [630, 346], [633, 354], [634, 356], [637, 357], [640, 359], [638, 357], [637, 353], [637, 348], [636, 347], [636, 343], [637, 341], [642, 340], [644, 341], [646, 342], [649, 345], [654, 348], [656, 349], [666, 349], [669, 347], [671, 343], [671, 341], [666, 331], [664, 329], [661, 327], [651, 322], [649, 320], [649, 311], [646, 310], [641, 310], [638, 309], [633, 304], [632, 296], [631, 292], [629, 288], [626, 284], [625, 281], [624, 278], [624, 268], [626, 264], [629, 257], [630, 255], [634, 250], [635, 248], [636, 246], [636, 239], [637, 238], [637, 232], [635, 228], [632, 215], [631, 210], [631, 204], [630, 201], [629, 198], [626, 193], [625, 190], [624, 184], [623, 182], [620, 179], [618, 178], [616, 178], [614, 177], [612, 175], [611, 173], [611, 169], [610, 166], [609, 164], [608, 162], [604, 155], [600, 151], [597, 150]]

def test_fbrs():
    with open(os.path.join(os.environ['PROJECT_DIR'], "tests", "fbrs_test.jpg"), "rb") as f:
        data = f.read()
        data = base64.b64encode(data).decode()

    pos_points = [(click[1][1], click[1][0]) for click in clicks if click[0]]
    neg_points = [(click[1][1], click[1][0]) for click in clicks if not click[0]]

    payload_dict = {"image": data, "pos_points": pos_points, "neg_points": neg_points, "threshold": 0.5}
    assert cvat_info() == cvat_info_answer
    if not os.path.exists(os.path.join(os.environ['PROJECT_DIR'], "fbrs_interactive_segmentation", "hrnet.json")):
        assert False, "hrnet.json doesn't exist!!"
    results = cvat_invoke(payload_dict)
    print(results)
    assert results == cvat_invoke_answer
