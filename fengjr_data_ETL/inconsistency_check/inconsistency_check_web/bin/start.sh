#!/bin/bush
. ~/.bashrc
timestamp(){
  date + "%Y-%m-%d %H:%M:%S"
}
log(){
  ts = `timestamp`
  echo  "$ts INFO: $@"
}
die(){
  ts = `timestamp`
  echo -e "\033[31m$ts ERR: $@ \033[0m"
  exit 1
}
pynumber=$(pyenv version | grep 3.6.5 | wc -l)
[[ $pynumber != 0 ]] || pyenv install 3.6.5
[[ $? == 0 ]] || die "安装python3.6.5失败.请手工安装."
pyenv rehash
pyenv virtualenv 3.6.5 inconsistency_check
pyenv activate inconsistency_check
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