# -*- coding: utf-8 -*-
"""
@Time     :2021/10/20 16:04
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
import typing

class LinkedList:
    """
    链表是指针串联起来的线性结构。
    单链表
    双链表
    循环链表：用于解决约瑟夫环问题

    """

    """
    203. 移除链表元素
    给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 
    Node.val == val 的节点，并返回 新的头节点 。
    
    思路解法：
        链表的移除时间复杂度为O(1),但是查找复杂度O(n)
        链表的移除操作，就是让节点指针指向下下个节点即可，特殊情况就是删除头结点，头结点没有前一个节点；
        两种操作方法：
            直接使用原来的链表删除
                删除头结点：将头结点向后移动一位；
                删除其余节点：将前一个节点执行下下个节点即可；
                这就需要两端逻辑进行处理
            设置一个虚拟节点进行删除
                虚拟节点目的是只用同一个逻辑进行处理；
    
    注：
        虚拟头结点dummp作用是，dummp.next指向head，避免每次判断head有没有指针指向；处理head节点和其他节点的操作逻辑相同；
        否则，删除第一个节点head = head.next；非首节点pre.next = cur.next
    """

    def removeElements_notVirtual(self, head: ListNode, val: int) -> ListNode:

        # 处理删除头结点的问题
        while (head != None and head.val == val):
            head = head.next
        # 判断是否为空head
        if head == None:
            return head
        # 处理删除其他节点的问题
        # pre = head # 不是val
        # cur = head.next
        # while(cur != None):
        #     if cur.val == val:
        #         pre.next = cur.next
        #     else:
        #         pre = cur
        #     cur = cur.next
        pre = head  # 不是val
        while (pre.next != None):
            if pre.next.val == val:
                pre.next = pre.next.next
            else:
                pre = pre.next
        return head




    def removeElements_Virtual(self, head: ListNode, val: int) -> ListNode:
        dummy_head = ListNode(next=head)
        cur = dummy_head
        while cur.next != None:
            if cur.next.val == val:# 如果是的话删除节点；
                cur.next = cur.next.next
            else:               # 如果不是的话，cur向后移动一位
                cur = cur.next
        return dummy_head.next

    """
    206. 反转链表
    给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
    思路解法：
        只需要改变next指针的指向就能翻转链表，翻转需要三个变量分别表示pre、cur、和保存cur.next；
        初始化pre=None
        初始化cur=head  # 然后从cur节点一个个的往后遍历，将cur指向pre
        while(cur):
            tmp = cur.next  存储下一个节点
            cur.next = pre
            pre = cur
            cur = tmp
    """
    def reverseList(self, head: ListNode) -> ListNode:
        tmp = ListNode()
        cur = head
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            # 更新 pre cur
            pre = cur
            cur = tmp
        return pre

    def reverseList2(self, head: ListNode) -> ListNode:
        """
        递归法：确定递归函数；确定终止条件；确定循环逻辑
        1. 递归函数
            传入pre和cur
        2. cur为空就返回
        3. 和迭代法一样
            递归函数传入None,head
                如果 cur存在就继续迭代，否则接跳出循环，return pre即可
                存储下一个
                cur.next 指向pre
                pre = tmp
                cur = pre
        :param head:
        :return:
        """
        def reverseList_recursion(pre, cur) -> ListNode:
            if not cur:
                return pre
            tmp = cur.next
            cur.next = pre
            return reverseList_recursion(cur, tmp)
        return reverseList_recursion(None, head)


    """
    24. 两两交换链表中的节点
    给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
    你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
    
    思路和算法：
        翻转链表的进阶版本；
        一定要画图，且要理解翻转过程； 虚拟头指向2节点，2节点指向1节点，1节点指向3节点；后边就是循环了；
        时间复杂度：O(n)；空间复杂度O(1)
        
    补充：
    无论是迭代法还是递归法：一定要明白迭代的逻辑，是哪些参数需要进行迭代；这样才能简洁的编写代码
    """
    def swapPairs(self, head: ListNode) -> ListNode:
        # 确定虚拟头结点，最后返回的时候用得着
        res = ListNode(next=head)
        # 需要 pre cur post三个节点进行翻转，循环需要判定当前节点，和下一个节点，所以预定义pre即可
        pre = res
        while pre.next and pre.next.next:
            cur = pre.next
            post = pre.next.next

            pre.next = post
            cur.next = post.next
            post.next = cur

            pre = cur
        return res.next

    """
    19. 删除链表的倒数第 N 个结点
    给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
    进阶：你能尝试使用一趟扫描实现吗？
    
    思路和算法：
        只能扫描一次  --》》》 双指针  将两个for循环缩减为1个
    """
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        res = ListNode(next=head)
        left = right = res

        while n != 0:
            right = right.next
            n -= 1
        while right.next != None:
            left = left.next
            right = right.next
        left.next = left.next.next
        return res.next


    """
    面试题 02.07. 链表相交
    给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
    图示两个链表在节点 c1 开始相交：
    
    思路和解法：
        交点是指针相同；
        得到两个链表的长度，将长链表移动到短链表齐平的位置。然后如果不为空依次向后移动，如果相等跳出，否则继续；
        时间复杂度 O(M+N)
        方法2：快慢法则
            因为有一个短的链表。只要其中一个链表走完，就去走另外一个链表。如果有交点，他们一定会在同一位置相遇
            （不是很理解这种方法）
    """
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        la = headA
        lb = headB
        lena = 0
        lenb = 0
        while la:
            lena +=1
            la = la.next
        while lb:
            lenb +=1
            lb = lb.next
        if lena > lenb:
            l = headA
            s = headB
            swap = lena - lenb
        else:
            l = headB
            s = headA
            swap = lenb - lena
        while swap != 0:
            l = l.next
            swap -= 1
        while l != None:
            if l == s:
                return l
            l = l.next
            s = s.next
        return None

    def getIntersectionNode_fastandslow(self, headA: ListNode, headB: ListNode) -> ListNode:
        cur_a, cur_b = headA,headB
        while cur_a != cur_b:
            cur_a = cur_a.next if cur_a else headB
            cur_b = cur_b.next if cur_b else headA
        return cur_a

    """
    环形链表
        判断是否存在环，首先明白什么是环。即通过不断next可以再次到达的节点；
        环形链表的表示：
            使用一个整数post表示链表尾部连接到环入口的位置（索引从0开始，-1表示没有）。
        如何判断是否有环：
            快慢指针法：fast每次移动两个节点，slow每次移动一个节点。如果存在环，他们就会相遇（一定是在环中相遇 不一定是环的入口）；
                原理：不管怎么追赶，总会存在fast在slow后面一位的情况，下次移动他们就会相遇。
            
    142. 环形链表 II
        给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
        为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
        说明：不允许修改给定的链表。
        进阶：
        你是否可以使用 O(1) 空间解决此题？
    思路和算法：
        考察是否存在环（快慢指针）；如果有环怎么查找环的入口（画图理解）；
            慢指针走过：x + y；快指针走过：x+y+n(y+z)
            又因为 2(x+y) = x + y+n(y+z)  -> 即x+y = n(y + z)
            环的入口就是求x的长度：x = (n-1)(y + z) + z；其中n为正整数，因为fast至少走一圈才能遇到slow；
            n=1 时 x=z表示fast转一圈就遇到slow；此时在head出发一个指针；在相遇点出发一个指针，他们相遇的地方就是起点；
            n>1 时 表示从相遇点出发的指针一定转了很多圈了，他们再次相遇同样式起点；
        
        编程逻辑：
            考虑不存在环的情况：while fast and fast.next
            在快慢指针的while循环中找到相等的点，使用if即可；
            在if中嵌套while寻找入口，一定能找到，返回即可
            如果最外层while找不到就返回None
    """
    def detectCycle(self, head: ListNode) -> ListNode:
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                new_f = head
                new_s = slow
                while new_f != new_s:
                    new_f = new_f.next
                    new_s = new_s.next
                return new_s
        return None
        # while fast != slow:
        #     fast = fast.next.next
        #     slow = fast.next
        # slow = head
        # while fast != slow:
        #     fast = fast.next
        #     slow = slow.next
        # return fast


## 146. LRU 缓存机制
class LeetCodeAboutLinkedList:
    """
    146. LRU 缓存机制
    运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
    实现 LRUCache 类：

    LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
    int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
    void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

    进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？
    思路和算法：
        因为需要存储key,value，所以首先考虑使用字典来存储key-value结构，这样对查找操作时间复杂度为O(1)
        但是字典本身无序，所以需要一个类似于队列的结构来记录访问的先后顺序，这个队列需要支持：
            从末尾加入一项；去除最前端一项；将队列中某一项移动至末尾；
        考虑列表结构，列表有append(),pop()都是O(1)时间复杂度，但是把列表中已有项移动至末尾，常数时间内无法挑选出来移动到末尾。
        考虑单链表，使用单链表实现哈希表的结构：{key:ListNode(value)}即键对应节点的地址，值是value。对于链表查找节点可以常数时间内；但是移动到末尾则需要从头遍历到该节点，才能保证链表不断，时间复杂度也为O(n)
        解决移动到末尾这个问题，需要使用双链表结构。
    """
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.hashmap = {}   # key = key; value: 双端链表中的节点
            self.head = LinkedNodeDouble(0)
            self.tail = LinkedNodeDouble(0)
            self.head.next = self.tail
            self.tail.prev = self.head

        def move_to_tail(self, key):
            """
            get和put都需要将双向链表中的节点移动到末尾
            :param key:
            :return:
            """
            node = self.hashmap[key]
            node.prev.next = node.next
            node.next.prev = node.prev

            node.prev = self.tail.prev
            node.next = self.tail
            self.tail.prev.next = node
            self.tail.prev = node

        def get(self, key):
            if key in self.hashmap:
                self.move_to_tail(key)
            node = self.hashmap.get(key, -1)
            if node == -1:
                return node
            else:
                return node.val

        def put(self, key, value):
            if key in self.hashmap:
                self.hashmap[key].value = value
                self.move_to_tail(key)
            else:
                if len(self.hashmap) == self.capacity:
                    self.hashmap.pop(self.head.next.val)
                    self.head.next = self.head.next.next
                    self.head.next.prev = self.head
                new = LinkedNodeDouble(value)
                self.hashmap[key] = new
                new.prev = self.tail.prev
                new.prev = self.tail
                self.tail.prev.next = new
                self.tail.prev = new

## 单链表
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = None
class MyLinkedList:
    """
    链表的head和tail只是起到一个标志作用，空值，表示首尾；
    链表索引是从0开始的，0号数据即head.next
    链表不是随机存取结构，需要按续搜索。
    """
    def __init__(self):
        self.size = 0
        self.head = ListNode(0)

    def get(self, index:int)->int:
        if index < 0 or index >= self.size:
            return -1
        cur = self.head
        for _ in range(index+1): # 走一步到索引0的位置，左闭右开，所以要取index+1
            cur = cur.next
        return cur.val
    def addAtHead(self, val):
        self.addAtIndex(0, val)
    def addAtTail(self, val):
        self.addAtIndex(self.size, val)
    def addAtIndex(self, index, val):
        """
        在index节点前插入新节点，如果index=0新节点为头结点
        如果index等于链表长度，说明插在链表的结尾； # 链表从索引从0开始的，区分链表长度
        如果index大于链表长度，返回空
        :param index:
        :param val:
        :return:
        """
        if index > self.size:
            return
        if index < 0:
            index = 0
        self.size+=1
        # 找到需要添加节点的前一个节点
        pred = self.head
        for _ in range(index):
            pred = pred.next
        toAdd = ListNode(val) # 新建要插入的节点
        toAdd.next = pred.next
        pred.next = toAdd
    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        self.size -=1
        pred = self.head
        for _ in range(index):
            pred = pred.next
        pred.next = pred.next.next

## 双链表
class LinkedNodeDouble:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None
class MyLinkedListDouble:
    def __init__(self):
        self.size = 0
        self.head, self.tail = LinkedNodeDouble(0), LinkedNodeDouble(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, index):
        if index < 0 or index >= index:
            return -1

        # 判断离那边近一点，后端长度为size-index,前段长度为index+1
        if self.size - index > index + 1:
            cur = self.head
            for _ in range(index+1):
                cur = cur.next
        else:
            cur = self.tail
            for _ in range(self.size - index):
                cur = cur.prev
        return cur.val

    def addAtHead(self, val):
        pre, nex = self.head, self.head.next
        self.size += 1
        to_add = LinkedNodeDouble(val)
        pre.next = to_add
        to_add.prev = pre
        to_add.next = nex
        nex.prev = to_add

    def addAtTail(self, val):
        pre, nex = self.tail.prev, self.tail
        self.size += 1
        to_add= LinkedNodeDouble(val)
        pre.next = to_add
        to_add.prev = pre
        to_add.next = nex
        nex.prev = to_add

    def addAtIndex(self, index, val):
        if index > self.size:
            return
        if index < 0:
            index = 0

        if self.size - index > index + 1:
            pre = self.head
            for _ in range(index):
                pre = pre.next
            nex = pre.next
        else:
            nex = self.tail
            for _ in range(self.size - index):
                nex = nex.prev
            pre = nex.prev
        self.size += 1
        to_add = LinkedNodeDouble(val)
        pre.next = to_add
        to_add.prev = pre
        to_add.next = nex
        nex.prev = to_add

    def deleteAtIndex(self,index):
        """
                Delete the index-th node in the linked list, if the index is valid.
                """
        # if the index is invalid, do nothing
        if index < 0 or index >= self.size:
            return

        # find predecessor and successor of the node to be deleted
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next.next
        else:
            succ = self.tail
            for _ in range(self.size - index - 1):
                succ = succ.prev
            pred = succ.prev.prev

        # delete pred.next
        self.size -= 1
