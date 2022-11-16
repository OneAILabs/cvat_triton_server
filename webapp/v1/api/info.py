# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from flask import request, g

from . import Resource
from .. import schemas
from cvat_custom.api import cvat_info

class Info(Resource):

    def get(self):
        resp = cvat_info()
        return resp, 200, None

