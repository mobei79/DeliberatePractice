# -*- coding: utf-8 -*-
"""
@Time     :2021/9/1 14:29
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from __future__ import absolute_import
from __future__ import division
import os
from pprint import pprint
from datetime import datetime
from utils.log import logger
from settings import config


def read_options():
    conf = config['model']
    conf['output_dir'] = conf['base_output_dir'] + '/' + datetime.now().strftime('%Y%m%d%H%M')
    conf['path_logger_file'] = conf['output_dir']
    if not os.path.exists(conf['output_dir']):
        os.makedirs(conf['output_dir'])
    with open(conf['output_dir'] + '/config.txt', 'w') as out:
        pprint(conf, stream=out)

    logger.info('#' * 80)
    maxLen = max([len(ii) for ii in conf.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    logger.info('Arguments:')
    for keyPair in sorted(conf.items()):
        logger.info(fmtString % keyPair)
    logger.info('#' * 80)
    return conf
