# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from flask import request, g

from . import Resource
from .. import schemas
from cvat_custom.api import cvat_invoke

class Invoke(Resource):

    def post(self):
        results = cvat_invoke(g.json)
        return results, 200, None

