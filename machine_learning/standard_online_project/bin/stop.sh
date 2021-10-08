#!/bin/bash
pid=$(netstat -lnpt | grep 500 | awk '{print $NF}' | awk -F "/" '{print $1}')
if [[ ! -z $pid ]]     # -z判断变量的值是否为空
then
  kill pid
else
  echo "未发现应用启动实例."
fi
# awk [选项] ‘命令’ 操作文件  ；
#   选项'-F' 指定分隔符，分隔符用""引起来; awk -F "/" “{print $1}” /etc/passwd ;以":"为分隔符打印/etc/passwd文件的第一例内容;
# awk '{print $0}' 1.txt;   $0表示逐行去文件内容并打印，$0保存当前行内容
#                           $" addtext" 表示逐行读取文件，并在每行结尾打印“addtext”;
#                           $1 表示打印第一列的内容; 默认空白分隔符
#                           ；
# awk `{print}` 1.txt 逐行打印1.txt中的内容；
# NF表示一行有做少个单次，$NF表示其中的最后一个





#type [[ 可以查看说明文档