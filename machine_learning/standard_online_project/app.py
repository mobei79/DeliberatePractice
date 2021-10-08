# -*- coding: utf-8 -*-
"""
@Time     :2021/9/1 13:52
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import json
# from minerva.model.predictor import Predictor
from minerva.options import read_options
from flask import Flask, Blueprint, request, jsonify
from utils.log import logger
app = Flask(__name__)

options = read_options()
# predictor = Predictor(options)

print(options)
# @app.route('/predict/entity', methods=['POST'])
# def predict_entity():
#     data = request.get_data()
#     json_data = json.loads(data.decode("utf-8"))
#     logger.info("request:{}".format(json_data))
#     data = json_data.get('data')
#     results = predictor.predict(data)
#     logger.info("response:{}".format(results))
#     response = jsonify({"results":results})
#     return response