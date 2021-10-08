#!/bin/bash
. ~/.bashrc
timestamp(){
  date +'%Y-%m-%d %H:%M:%S'
}
log(){
  ts=`timestamp`          #``和$(ts)都是用来做命令替换的类似变量替换，就是重命名，先执行完引号内的命令行，然后将结果替换出来，重命名得到新的命令行
  echo "$ts INFO: $@"     # $ts是引用变量；echo可以直接跟$TS也可以跟“双引号阔括起来的内容”；$@表示传递给函数的所有参数

}
die(){
  ts=`timestamp`
  echo -e "\033[31m$ts ERR: $@ \033[0m"      # echo -e表示激活转义字符
  exit 1            #退出当前程序
}
pynumber=$(pyenv versions | grep 3.6.7 | wc -l)
[[ $pynumber !=0 ]] || pyenv install 3.6.7              # command1 && command2 左边命令执行完之后执行右边命令 || 如果左边的命令没执行成功就执行右边命令
[[ $? == 0 ]] die "安装python3.6.7失败.请手工安装."        # $? 上个命令的退出状态，或函数的返回值。
pyenv rehash
pyenv virtualenv 3.6.7 fengjr-entity-prediction
pyenv activate fengjr-entity-prediction
# cd "$(dirname "$0")"/..
echo $PYTHON_PROJECT_ENV
if [[ $PYTHON_PROJECT_ENV == 'prod' ]]
then
  echo "========installing prod requirements_prod.txt======"
  pip install -r requirements_env/requirements_prod.txt -i https://mirrors.aliyun.com/pypi/simple/
else
  echo "========installing test requirements_test.txt======"
  pip install -r requirements_env/requirements_test.txt -i https://mirrors.aliyun.com/pypi/simple/
fi
# cd /export/server/fengjr-entity-prediction
nohup gunicorn -w 4 -b 0.0.0.0:5000 app:app >/dev/null 2>&1 &









# shell
#环境变量设置：
#进入当前用户主目录：cd ~ 或者 cd$HOME 或者 cd ~/
#进入环境变量配置文件： vim .bashrc 或者.bashrc
#
#shell 函数
#timestamp(){
#    date +'%Y-%m-%d %H:%M:%S'
#}
