# -*- coding: utf-8 -*-

###
### DO NOT CHANGE THIS FILE
### 
### The code is auto generated, your change will be overwritten by 
### code generating.
###
from __future__ import absolute_import

from .api.info import Info
from .api.invoke import Invoke

routes = [
    dict(resource=Info, urls=['/cvat_info'], endpoint='info'),
    dict(resource=Invoke, urls=['/cvat_invoke'], endpoint='invoke'),
]
