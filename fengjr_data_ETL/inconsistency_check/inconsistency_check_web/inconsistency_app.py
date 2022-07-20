# -*- coding: utf-8 -*-
"""
@Time     :2022/3/14 15:27
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import json
from flask import Flask, Blueprint, request, jsonify
from utils.log import logger
from inconsistency_check_web_file.service import query_diff
app = Flask(__name__)

@app.route('/query/inconsistency', methods=['POST'])
def query_inconsistency():
    req_data = request.get_data()
    data = json.loads(req_data.decode("utf-8"))
    logger.info("request:{}".format(data))
    results = query_diff(data['id_card'], data['prop'], data['income_tm_intervel'],
                         pct=float(data['percent']) if data['percent'] else None,
                         sim_fn=data['sim_method'] if data['sim_method'] else 'equal')
    logger.info("response={}".format(results))
    return jsonify(results)

if  __name__ == '__main__':
    app.run(host='0.0.0.0')