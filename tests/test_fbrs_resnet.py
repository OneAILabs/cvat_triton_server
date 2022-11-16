import os
import base64
from fbrs_interactive_segmentation.api import cvat_info, cvat_invoke
from fbrs_interactive_segmentation.isegm.inference.clicker import Click

clicks = [Click(is_positive=True, coords=(201, 605)), Click(is_positive=True, coords=(320, 623)), Click(is_positive=False, coords=(323, 658)), Click(is_positive=True, coords=(341, 658)), Click(is_positive=True, coords=(296, 615)), Click(is_positive=True, coords=(170, 574)), Click(is_positive=True, coords=(200, 622)), Click(is_positive=True, coords=(281, 570)), Click(is_positive=False, coords=(289, 549)), Click(is_positive=False, coords=(290, 548)), Click(is_positive=False, coords=(289, 548)), Click(is_positive=False, coords=(290, 547)), Click(is_positive=False, coords=(290, 549)), Click(is_positive=False, coords=(288, 550)), Click(is_positive=False, coords=(291, 548)), Click(is_positive=False, coords=(288, 549)), Click(is_positive=False, coords=(289, 550)), Click(is_positive=False, coords=(291, 549)), Click(is_positive=False, coords=(289, 547)), Click(is_positive=False, coords=(290, 546))]

cvat_info_answer = {"framework":"pytorch","spec": None,"type": "interactor","description": "f-BRS interactive segmentation"}

cvat_invoke_answer = [[587, 152], [579, 156], [576, 158], [572, 161], [566, 167], [564, 170], [563, 173], [561, 176], [563, 177], [574, 177], [576, 178], [577, 179], [578, 181], [579, 183], [579, 188], [582, 193], [582, 201], [584, 210], [583, 214], [580, 220], [573, 230], [572, 232], [571, 237], [566, 242], [566, 245], [568, 247], [570, 248], [572, 248], [574, 249], [576, 251], [577, 253], [577, 260], [576, 263], [572, 271], [567, 276], [565, 279], [562, 285], [561, 288], [560, 294], [561, 303], [566, 303], [570, 299], [571, 297], [571, 295], [572, 293], [573, 291], [580, 290], [583, 292], [586, 289], [590, 288], [594, 288], [597, 289], [602, 294], [605, 299], [608, 306], [610, 309], [614, 313], [615, 315], [615, 317], [612, 320], [613, 330], [627, 331], [630, 334], [631, 337], [631, 341], [634, 346], [634, 349], [634, 347], [635, 344], [638, 341], [644, 341], [648, 342], [652, 344], [656, 345], [660, 345], [663, 344], [665, 343], [665, 336], [661, 328], [658, 325], [655, 324], [653, 323], [652, 322], [652, 321], [650, 319], [649, 317], [649, 313], [647, 311], [644, 310], [639, 310], [636, 309], [632, 305], [631, 303], [631, 291], [630, 289], [622, 281], [621, 279], [621, 275], [622, 271], [626, 264], [627, 259], [628, 256], [631, 251], [634, 249], [635, 247], [636, 244], [636, 235], [634, 231], [634, 221], [632, 215], [632, 207], [631, 203], [629, 198], [625, 191], [625, 188], [623, 183], [621, 181], [617, 181], [615, 180], [613, 178], [612, 176], [611, 170], [608, 163], [603, 155], [600, 152]]

def test_fbrs():
    with open(os.path.join(os.environ['PROJECT_DIR'], "tests", "fbrs_test.jpg"), "rb") as f:
        data = f.read()
        data = base64.b64encode(data).decode()

    pos_points = [(click[1][1], click[1][0]) for click in clicks if click[0]]
    neg_points = [(click[1][1], click[1][0]) for click in clicks if not click[0]]

    payload_dict = {"image": data, "pos_points": pos_points, "neg_points": neg_points, "threshold": 0.5}
    assert cvat_info() == cvat_info_answer
    if not os.path.exists(os.path.join(os.environ['PROJECT_DIR'], "fbrs_interactive_segmentation", "resnet.json")):
        assert False, "resnet.json doesn't exist!!"
    results = cvat_invoke(payload_dict)
    print(results)
    assert results == cvat_invoke_answer
