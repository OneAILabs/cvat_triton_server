# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np

def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index

