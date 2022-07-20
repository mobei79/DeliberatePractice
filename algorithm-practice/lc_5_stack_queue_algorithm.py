# -*- coding: utf-8 -*-
"""
@Time     :2021/11/11 15:40
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from collections import Counter
class Solution:
    """347. 前 K 个高频元素
    给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
    示例 1:
    输入: nums = [1,1,1,2,2,3], k = 2
    输出: [1,2]
    示例 2:
    输入: nums = [1], k = 1
    输出: [1]
    提示：
    1 <= nums.length <= 105
    k 的取值范围是 [1, 数组中不相同的元素的个数]
    题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的

    思路和解法：
        直接想法：统计频率，排序取前k个；【小数据集没问题，大数据集耗时】，
            时间复杂度：构建哈希表O(N)，最坏情况有N个不同的次数的值，排序的话复杂度Nlog(N)；总的时间复杂度O(N)+Nlog(n)=Nlog(n)

        本题难点：控制时间复杂度，直接法是O(Nlog(N))
        方法1：hash表+堆 - O(Nlog(k))
            遍历整个数组，使用哈希表记录每个数字出现的次数；
            建立小顶堆，遍历哈希表：
                如果堆元素小于k，直接插入；
                如果堆元素等于k，比较堆顶元素和当前值大小；
                    如果堆顶小于当前值：弹出堆顶；将当前值插入堆中；
                    如果堆顶大于当前值：直接舍弃当前值
            时间复杂度：O(Nlog(k))。堆大小为k，每次堆操作耗时O(log(k))。
            空间复杂度：hash表O(n),堆O(k)。总共O(N)
    """
    # hash + 小顶堆 + 位操作
    def topKFrequent_bit(self, nums, k) :
        def sift_up(heap, child):
            val = heap[child]
            # << 左移动位运算符， 运算数的各二进位全部左移若干位，高位丢去，
            while child>>1 >0 and val[1] < heap[child>>1][1]: # 如果当前频次下余父节点频次
                heap[child] = heap[child>>1]
                child >>=1
            heap[child] = val
        def sift_down(heap, root, k): # 如果新的根节点》子节点就下沉
            val = heap[root] #
            while root<<1 < k:
                child = root<<1
                # 选取左右孩子中小的与父节点交换
                if child | 1 < k and heap[child | 1][1] < heap[child][1]:
                    child |= 1
                # 如果子节点<新节点,交换,如果已经有序break
                if heap[child][1] < val[1]:
                    heap[root] = heap[child]
                    root = child
                else:
                    break
            heap[root] = val

        from collections import Counter
        dic = Counter(nums)
        stat = list(dic.items())
        heap = [(0,0)]

        # 构建规模k+1的对，新元素加入堆尾部，上浮
        for i in range(k):
            heap.append(stat[i])
            sift_up(heap, len(heap)-1)
        # 维护规模k+1的堆，如果新元素大于堆顶，入堆，并下沉。 【小顶堆】
        for i in range(k, len(stat)):
            if stat[i][1] > heap[1][1]:
                heap[1] = stat[i]
                sift_down(heap, 1, k+1)
        return [item[0] for item in heap[1:]]

    # collectionsCounter 作弊 O(nlogn) O(n)
    def topKFrequent_collectionsCounter(self, nums, k):
        import collections
        counter = collections.Counter()
        return [item[0] for item in counter.most_common(k)]

    # # hash + 堆排序 不清楚函数的实现
    # def topKFrequent_collectionsCounter(self, nums, k):
    #     # count = Counter(nums)
    #     # heap = [(val, key) for key,val in count.items()]
    #     # return [item[1] for item in heapq.nlargest(k, heap)]
    #     count = Counter(nums)
    #     h = []
    #     for k,v in count.items():
    #         heapq.heappush(h, (val, key))
    #         if len(h) > k:
    #             heapq.heappop(h)
    #     return [x[1] for x in h]

    # hash + 小顶堆 + 常规
    def topKFrequent_util(self, nums, k) :
        """
        使用collection.Counter() 构建hash表
        前k个直接插入堆；
        k-len(num)需要比较判断；
        入堆：插入最后，然后上浮
        出堆：去掉顶部，替换为最后一个元素，然后下沉；
        """
        def sift_up(heap, child):
            """
            heap是append后的数组，新节点在最后一个，即index = child = len(heap)-1
            应该是和父节点进行比较，如果小于父节点，就交换；
                这里使用temp临时存储新节点；不断和父节点比较
            child的父节点：(child-1)/2
            """
            temp = heap[child]
            while child > 0 and (child-1)//2 > 0 and heap[(child - 1)//2][1] > temp[1]: # 父节点存在，且父节点大于子节点
                heap[child] = heap[(child - 1)//2] # 将父节点值 给子节点
                child = (child-1)//2
            heap[child] = temp
        def sift_down(heap, root, k): # 如果新的根节点》子节点就下沉
            """
            子节点为：2i+1和2i+2
            找出子节点中较小的那个；和root比较，如果root大，则替换；否则不变
            """
            val = heap[root]
            while root < k:
                child = root*2+1
                if child < k :
                    min_index = child
                    if child+1 <k:
                        if heap[child+1][1] < heap[child][1]:
                            min_index = child+1
                    if heap[root][1] > heap[min_index][1]:
                        heap[root][1], heap[min_index][1] = heap[min_index][1], heap[root][1]
                    else:
                        break
                    root = min_index


        from collections import Counter
        dic = Counter(nums)
        stat = list(dic.items())
        heap = []

        # 构建规模k+1的对，新元素加入堆尾部，上浮
        for i in range(k):
            heap.append(stat[i])
            sift_up(heap, len(heap)-1)
        # 维护规模k+1的堆，如果新元素大于堆顶，入堆，并下沉。 【小顶堆】
        print(heap)
        for i in range(k, len(stat)):
            if stat[i][1] < heap[0][1]:
                heap[0] = stat[i] # 直接替换掉小顶堆的堆顶
                sift_down(heap, 0, k) # 下沉
        return [item[0] for item in heap[0:]]

    # 基于优化的快速排序
    def topKFrequent_quickSort_pro(self, nums, k):
        """
        基于快速排序，求出topk
        将数据分为两部分，并使得做部分每个值都不超过mid，右边部分都大于mid；
        判断右边部分和k的关系：
            len(right)>=k
        :param nums:
        :param k:
        :return:
        """

    #394. 字符串解码
    """ 394. 字符串解码
    给定一个经过编码的字符串，返回它解码后的字符串。
    编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
    你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
    此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
    示例 1：
    输入：s = "3[a]2[bc]"
    输出："aaabcbc"
    示例 2：
    输入：s = "3[a2[c]]"
    输出："accaccacc"
    示例 3：
    输入：s = "2[abc]3[cd]ef"
    输出："abcabccdcdcdef"
    示例 4：
    输入：s = "abc3[cd]xyz"
    输出："abccdcdcdxyz"   
    提示：
    1 <= s.length <= 30
    s 由小写英文字母、数字和方括号 '[]' 组成
    s 保证是一个 有效 的输入。
    s 中所有整数的取值范围为 [1, 300] 
    
    思路和解法：
        直接想法：遍历s，如果是数字就记录num，然后通过判断得到【】内部的字符串sub_str。进行拼接res+num*sub_str;
        
        本题难点：括号内嵌套括号，这就需要由内向外生成和拼接；栈的特性是先入后出。
        方法1： 构建辅助栈 stack，遍历s中每个字符c
            如果c是数字，转成数字multi，用于倍数计算
            如果c是字母，在res尾部添加c
            如果c是[时，将multi和res入栈，并分别置空置0；
                记录此【前的临时结果res入栈，用于发现]后拼接
                记录此【前的multi入栈，用于发现]后 倍乘
            如果c是】时，stack出栈，】拼接res = last_res + cur_multi*res
                last_res 是上个[ 到当前【的字符串，例如"3[a2[c]]" 中的 a；
                cur_multi是当前 [ 到 ] 内字符串的重复倍数，例如 "3[a2[c]]" 中的 2。
            返回res
        方法2： 递归法：将【】分别作为递归的开始与结束条件。
        遍历s中的每一个字符c；
            如果c是数字，转成num；
            如果是字母，在res尾部添加c
            如果c是【，进入递归，记录此时[]内字符串tmp和递归后的最新索引，res+num*tmp拼接；
            如果c是】，返回当前括号内记录的res字符串，与]的索引i，分会上层递归
    """
    def decodeString(self, s: str) -> str:
        stack,res,num = [],"",0
        for c in s:
            if c == '[':
                stack.append([num, res])
                num=0
                res=""
            elif c == ']': # 表示结束了，将返回stack中的num和res
                cur_num, pre_res = stack.pop()
                res = pre_res+cur_num*res
            elif  '0'<=c<='9':
                num = num*10 + int(c)
            else:
                res += c
        return res
    def decodeString_digui(self, s: str) -> str:
        def decodeString_digui_sub(s, index):
            """
            终止条件，遍历遇到“】”，
            返回值：返回当前括号中的字符串，以及字符串结束的位置；
            输入：字符串，即“【”开始的位置
            """
            if index == len(s) or s[index] == '[': # 如果查过字符串了，或者是]说明没有sub——string，直接返回就行
                return "", len(s)
            res = "" #本段内的字符串
            cur_num = 0 # 重复次数
            while index < len(s): # for i in range(index, len(s)): # 从给定的起点开始
                if '0' <= s[index] <= '9': # 统计重复次数
                    cur_num = cur_num*10 + int(s[index])
                elif s[index] == '[': # 如果匹配到[。要递归内部的数据，并返回内部字符串cur_res，和结束位置index 即]所在的位置
                    cur_res, index = decodeString_digui_sub(s, index+1)
                    res += cur_res*cur_num
                    cur_num = 0
                elif s[index] == ']': # 匹配到]直接返回字符串res，和当前位置；
                    return res,index
                else:
                    res += s[index]
                index +=1
            return res,index

        return decodeString_digui_sub(s, 0)[0]
        #     res, num = "", 0
        #     while index < len(s):
        #         if '0' <= s[index] <= '9':
        #             num = num * 10 + int(s[index])
        #         elif s[index] == '[':
        #             i, tmp = decodeString_digui_sub(s, index+1)
        #             res += num*tmp
        #             num = 0
        #         elif s[index] == ']':
        #             return index, res
        #         index += 1
        #     return res
        # return decodeString_digui_sub(s, 0)



class StactQueue:
    """
    了解java中常见的stack queue容器；
    栈实现队列；队列实现栈；
    栈经典题型：
        系统中括号 花括号 路径 递归等都是 栈实现的。
        括号匹配问题：
        字符串去除相邻重复项；
        逆波兰表达式
    队列经典题型：
        滑动窗口中的最大值？
        前k高频 ： 优先级队列（其实就是堆）

    """

class MyQueue:
    """
    232. 用栈实现队列
    请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

    实现 MyQueue 类：

        void push(int x) 将元素 x 推到队列的末尾
        int pop() 从队列的开头移除并返回元素
        int peek() 返回队列开头的元素
        boolean empty() 如果队列为空，返回 true ；否则，返回 false

    思路：
        python中没有队列，使用数组实现即可。
    """

    def __init__(self):
        """
        in主要负责push，out主要负责pop
        """
        self.stack_in = []
        self.stack_out = []

    def push(self, x: int) -> None:
        """
        有新元素进来，就往in里面push
        """
        self.stack_in.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return None

        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

from collections import deque
class MyStack:

    def __init__(self):
        """
        Python普通的Queue或SimpleQueue没有类似于peek的功能
        也无法用索引访问，在实现top的时候较为困难。

        用list可以，但是在使用pop(0)的时候时间复杂度为O(n)
        因此这里使用双向队列，我们保证只执行popleft()和append()，因为deque可以用索引访问，可以实现和peek相似的功能

        in - 存所有数据
        out - 仅在pop的时候会用到
        """
        self.queue_in = deque()
        self.queue_out = deque()

    def push(self, x: int) -> None:
        """
        直接append即可
        """
        self.queue_in.append(x)

    def pop(self) -> int:
        """
        1. 首先确认不空
        2. 因为队列的特殊性，FIFO，所以我们只有在pop()的时候才会使用queue_out
        3. 先把queue_in中的所有元素（除了最后一个），依次出列放进queue_out
        4. 交换in和out，此时out里只有一个元素
        5. 把out中的pop出来，即是原队列的最后一个

        tip：这不能像栈实现队列一样，因为另一个queue也是FIFO，如果执行pop()它不能像
        stack一样从另一个pop()，所以干脆in只用来存数据，pop()的时候两个进行交换
        """
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())

        self.queue_in, self.queue_out = self.queue_out, self.queue_in  # 交换in和out，这也是为啥in只用来存
        return self.queue_out.popleft()

    def top(self) -> int:
        """
        1. 首先确认不空
        2. 我们仅有in会存放数据，所以返回第一个即可
        """
        if self.empty():
            return None

        return self.queue_in[-1]

    def empty(self) -> bool:
        """
        因为只有in存了数据，只要判断in是不是有数即可
        """
        return len(self.queue_in) == 0

def peek(self) -> int:
    """
    Get the front element.
    """
    ans = self.pop()
    self.stack_out.append(ans)
    return ans

def empty(self) -> bool:
    """
    只要in或者out有元素，说明队列不为空
    """
    return not (self.stack_in or self.stack_out)


if __name__ == "__main__":
    # a = 1
    # import sys
    # len = sys.getsizeof(a)
    # print(2**30)
    # print(len)
    # print(sys.getsizeof(2**30))
    # print(sys.getsizeof(2 ** 30-1))
    # print("*"*20)
    # print(sys.getsizeof(0.001))
    # print(sys.getsizeof(2**25+0.01))
    #
    # s='9'
    # print(ord(s)>ord('0'))

    exam = Solution()

    # a = exam.decodeString_digui("3[a2[c]]")
    # print(a)

    numss = [7,7,7,7,7, 1, 1, 1, 2, 2, 3]
    k = 2
    rst = exam.topKFrequent_util(numss, k)
    print(rst)
    # import collections
    # a = [1,3,3,77,5,5,6,6,6]
    # stat = collections.Counter(a).items()
    # print(list(stat))