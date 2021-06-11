# -*- coding: utf-8 -*-
"""
@Time     :2020/11/24 15:59
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
# 生成10万条 1-10000之间的随机数 文件
import math
import codecs
import requests
import random
import time
import threading

class MyThread(threading.Thread):
    def __init__(self,path:str,key:str,start_index:int,end_index:int,w_file):
        super().__init__()
        self.path = path
        self.key = key
        self.start_index = start_index
        self.end_index = end_index
        self.w_file = w_file
    def run(self):
        with open(self.path,'r') as f:
            global count
            global info_list
            envUrl = "http://p2passet_admin_finance.local.fengjr.inc/api/getAbleWithDrawByLoanId?loanId="  # 请求URL地址
            for line in f.readlines()[self.start_index:self.end_index]:
                line = line.strip()
                get_data_url = envUrl + line
                try:
                    # print(get_data_url)
                    res = requests.get(get_data_url)
                    if res.content:
                        res = res.json()
                        data = res.get('ableWithdraw', -1)
                    else:
                        data = "0"
                    count += 1
                    print(f'\r当前处理了{count}行',end='')
                    self.w_file.write('{user_id},{datas}\n'.format(user_id=line, datas=data))
                except Exception as e:
                    self.w_file.write("error item : {user_id}\n".format(user_id=line))
                    # time.sleep(6)
                    print(e)
                    continue



                # if line.find(self.key) != -1:
                #     count += 1
                #     key_list.append(line)
                # print(f'\r已经找到{count}个',end='')
# 全局变量
count = 0
key_list = []
file_length = 0


def main(path:str,thread_num:int,key:str):
    # 获取文件行数
    global file_length
    # for file_length,line in enumerate(open(path,'r')):
    #     pass
    # file_length += 1
    file_length = len(open(path,'r').readlines())
    # 文件切割，按照线程数目切割，只需要切线程次数-1次
    cut_size = math.ceil(file_length/thread_num)
    # 线程列表
    thread_list = []
    with open('rst_loanid_amount_map_100k.csv','w') as w_file:

        for i in range(thread_num):

            # 传入每个线程检索的起始和终止位置
            thread = MyThread(path,key,cut_size * i,cut_size * (i+1),w_file)
            # 开始线程
            thread.start()
            thread_list.append(thread)
        # 等待所有线程结束
        for th in thread_list:
            th.join()
# with open("random_10k.txt",'w+') as f:
#     for i in range(1000000):
#         f.write(str(random.randint(1,10000))+'\n')


def find(key: str,file_path):
    with open(file_path,'r') as r_file:
        ket_list = []
        count = 0
        print("正在检索数据中>>>>>>>>")
        for line in r_file.readlines():
            line = line.strip()
            if line.find(key) != -1:
                count += 1
                ket_list.append(line)
            print(f"\r已经找到{count}个。",end='')
        print("\n检索完毕")
        print(ket_list)


def get_loanid_amount_map():
    """
    直接调用    线程内容
    :return:
    """
    envUrl = "http://p2passet_admin_finance.local.fengjr.inc/api/getAbleWithDrawByLoanId?loanId="     # 请求URL地址
    read_csv_path="loanids_100k.txt"
    write_csv_file="shanchu_loanid_amount_map.csv"
    with open(read_csv_path) as r_file:
        target_ids = r_file.readlines()
    # print(target_ids)
    print("本次需要查询的数据条数为 ：",str(len(target_ids)))
    #query interface
    item_cnt = 0
    with codecs.open(write_csv_file, 'a', 'utf-8') as f:
        # f.write('loanId,ableWithdraw\n')
        for item in target_ids:
            item = item.strip()
            # if item_cnt<1000:
            get_data_url = envUrl+item
            # print(get_data_url)
            try:
                # print(get_data_url)
                res = requests.get(get_data_url)
                if res.content:
                    res = res.json()
                    data = res.get('ableWithdraw',-1)
                else:
                    data="0"
                item_cnt += 1
                # print(item_cnt)
                f.write('{user_id},{datas}\n'.format(user_id=item,datas=data))
            except Exception as e:
                f.write("error item : {user_id}\n".format(user_id=item))
                # time.sleep(6)
                print(e)
                continue
                # raise e

    print(f"\r当前已经请求完成的条数：{item_cnt}",end='')

def main_insantiation():
    print('开始实例化threading.Thread')
    thread_1 = threading.Thread(target=get_loanid_amount_map,name='th1')
    thread_2 = threading.Thread(target=get_loanid_amount_map,name='th2')
    thread_1.start()
    thread_2.start()
    print('main thread has ended!')

# 创建锁
lock = threading.Lock()

# 全局变量
GLOBAL_RESOURCE = [None] * 5
def test_thread(para='hi',sleep=5):
    """线程运行函数"""
    lock.acquire()
    global GLOBAL_RESOURCE
    for i in range(len(GLOBAL_RESOURCE)):
        GLOBAL_RESOURCE[i] = para
        time.sleep(sleep)
    print(f"修改全局变量为：{GLOBAL_RESOURCE}")
    lock.release()
    # time.sleep(sleep)
    # print(f'para:{para}')

def main_test_thread():
    """
    join(timeout=None)：让当前调用者线程（一般是主线程）等待，知道线程结束，timeout参数以秒为单位，用于设置超时时间
    如果要判断线程是否超时，只能通过现成的is_alive方法进行哦安短。
    :return:
    """
    print("开始实例化threading.Thread")
    thread_hi = threading.Thread(target=test_thread, args=("hi", 1))
    thread_hello = threading.Thread(target=test_thread, args=('hello', 1))
    # 启动线程
    thread_hi.start()
    thread_hello.start()
    time.sleep(2)
    print('马上执行join方法了')
    # 执行join方法会阻塞调用线程（主线程），直到调用join方法的线程（thread_hi）结束
    thread_hi.join()
    print('线程thread_hi已结束')
    # 这里不会阻塞主线程，因为运行到这里的时候，线程thread_hello已经运行结束了
    thread_hello.join()
    print('Main thread has ended!')

if __name__=='__main__':
    """
    直接运行
    """
    # get_loanid_amount_map()
    """
    通过实例化threading.Thread类创建线程
    """
    # main_test_thread()
    main_insantiation()
    """
    通过集成threading.Thread类的子类创建线程
    """
    # start_tm_multi = time.time()
    main('loanids_100k.txt',4,'7979')
    # print(f'\n检索完毕，共检索{file_length}条数据')
    # print(key_list)
    # print('耗时',time.time()-start_tm_multi)