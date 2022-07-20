#!/bin/bash
pid=$(netstat -lnpt | grep 5000 | awk '{print $NF}' | awk -F "/" '{print $1}')
if [[ ! -z $pid ]]
then
    kill $pid
else
    echo "未发现应用启动实例."
fi