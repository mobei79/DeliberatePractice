"""

1号店贿赂问题：
店铺数量n，编号1-n
人的数量吗，编号1-m
每个人有自己投票的店铺p，以及改投1号店的报价x
返回想让1号店铺人气最高，最少需要花费的金额
1 《= p,n,m <= 3000
1 <= x <= 10^9
"""

pass

"""
做笔试题第一要务是看数据量，只要最终计算量在10^8以内。
举例说明效果会很好。

拉人过程：
遍历所有可能称为人气王的人员数量；
    如果有比1号店人数多的店，就把花销最小的人拉过来；
    否则就在所有人中补齐人数；
只要选择其中能成为人气王，且金额最小的店；

涉及到的几个阈值：
    不是每个人数目都能成为人气王，比如自身的人数+需要拉人的人数大于目标人数，就说明这个人数无法成为人气王；
    人数多少和需要花费的金额没有单调关系。

思考过程：
    二分答案方向，但是人数和花费的金额不存在单调关系，
        比如2个人的花费小于3个人的花费。如果有3号店分别是（3， 100）；4号店分别是（100,200），
        如果想让1号店以两个人称为人气王就需要消减人数为2的店的人气。至少花费103；
        但是如果以三个人称为人气王，只需要找花费最少的人就可以实现。
    数据量分析：笔试题的时间上限是10^8
        店铺和人都在3000内，不断尝试让1号店称为人气王的人数和金额；最多3000规模；只要每次的复杂度在3000内，就一定能过，即暴力法。
        
    
    
"""
def min_cost_1(n, m, arr):
    """
    :param n: 店铺数量
    :param m: 人的数量
    :param arr: arr[][] 每个人投给德店，以及回贿赂金额； 如果本来是1号店，后面的金额就没用了，
    :return:
    1. 统计每个店铺支持的人数，所以需要一个统计词频的数字，下表是1-n;
    2. 需不需要贿赂need_change，它本身是不是人气王
    3. 否就遍历所有人数，计算需要的金额process
    """
    cnts = []*(m+1)
    for i in arr:
        cnts[arr[0]] += 1

    need_change = False # 默认不需要贿赂
    for i in range(2,len(cnts)):
        if cnts[i] > cnts[1]:
            need_change = True
            break
    if not need_change: # 如果本身是人气王，返回0
        return 0

    """
    定好一个人数，用什么样的策略进行贿赂花钱最少；
        开始，我们知道所有人需要转投的钱数，（其中有人已经投了1号，这些人的花费其实可以不用）      
        进行一些数据处理：
            需要把所有人根据投的钱数排序，由少到多；排序后人员也重新编号了，因为每个人的下表变了；原来是0号人，投3号店，需要10元；排序后是3号人，投5号店，需要1元；按金额排序；（其实我们并不在乎是几号人，只需要知道投的店以及金额就行）
            对每个店铺建立支持者队列，即每个店的按照金额进行排序；【这里按照重新排序的序号】
        现在人数大于等于目标人数x的店，都需要把金额最少的人转投过来，如果人数不满就选取金额最少的人（不能选择已经选择过得人，这就需要一个标志位bool），且最终满足人数条件，
    """
    sorted(arr, key= lambda item:item[1]) #人也进行了重新编号；
    shops = []*n
    for i in range(n):
        shops[arr[i][0]].append(arr[i][1])

    return process(arr, 0, n, bool*m)


def process():
    """
    :return:
    """
    pass

cnts = [0]*4
arr = [[1,3],[3,8],[2,9]]
print(arr)
for i in arr:
    print(i)
    cnts[i[0]]+=1

print(cnts)
print(sorted(arr,key=lambda item:item[1]))
shops = [[]*4]
print(shops)
for i in range(len(arr)):
    print(arr[i][1])
    shops[arr[i][0]].append(arr[i][1])