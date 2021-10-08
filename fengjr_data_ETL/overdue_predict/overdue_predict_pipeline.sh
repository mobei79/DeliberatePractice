#!/bin/bash

cd /home/supdev/yongguan/overdue_predict/predict_scheduler
nohup /usr/bin/python2.7 daily_run.py 2>&1 &
