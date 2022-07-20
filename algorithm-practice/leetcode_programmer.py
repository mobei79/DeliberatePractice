# -*- coding: utf-8 -*-
"""
@Time     :2021/10/27 15:24
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
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

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hashmap = {}
        # 新建两个节点 head tail
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化链表 head <-> tail
        self.head.next = self.tail
        self.tail.prev = self.head
    def move_to_tail(self, key):
        """
        get和put操作都需要将双向链表中的某个节点移动到末尾

        :param key:
        :return:
        """
        # 先将哈希表key指向的节点拎出来，为了简洁起名node
        #      hashmap[key]                               hashmap[key]
        #           |                                          |
        #           V              -->                         V
        # prev <-> node <-> next         pre <-> next   ...   node
        node = self.hashmap[key]
        node.prev.next = node.next
        node.next.prev = node.prev
        # 之后将node插入到尾节点前
        #                 hashmap[key]                 hashmap[key]
        #                      |                            |
        #                      V        -->                 V
        # prev <-> tail  ...  node                prev <-> node <-> tail
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key in self.hashmap:
            self.move_to_tail(key)
        res = self.hashmap.get(key, -1)
        if res == -1:
            return res
        else:
            return res.value

    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            self.hashmap[key].value = value
            self.move_to_tail(key)
        else:
            if len(self.hashmap) == self.capacity:
                self.hashmap.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            new = ListNode(key, value)
            self.hashmap[key] = new
            new.prev = self.tail.prev
            new.next = self.tail
            self.tail.prev.next = new
            self.tail.prev = new



class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class Solution:
    """
    3. 无重复字符的最长子串
    给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
    """
    def lengthOfLongestSubstring(self, s) -> int:

        s = "bbbbb"
        occ = set()
        n = len(s)
        l = 0
        max = 0
        for r in range(n):
            if r == 0:
                max +=1
            if s[r] in occ:
                while s[l] != s[r]:
                    occ.remove(s[l])
                    l+=1

                l +=1
            if l < r and r - l + 1 > max:
                max = r - l + 1
            occ.add(s[r])
        print(max)
        return max

        #
        #
        # left = 0
        # max = 0
        # for right in range(n):
        #     if s[right] not in occ:
        #         occ.add(s[right])
        #         if right - left + 1 > max:
        #             max = right - left + 1
        #     else:
        #         while s[left] != s[right]:
        #             left+=1
        #             occ.remove(s[left])
        #         left +=1
        #         if right - left + 1 > max:
        #             max = right - left + 1


class DoublePoint:
    """
    27. 移除元素 - 原址移动元素
    给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
    元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
    思路和算法：
        数组是连续存储的，不能直接移除元素。有需要O(1)的空间限制
    """
    def removeElement(self, nums, val):
        left = 0
        right = 0
        for right in range(len(nums)):
            while nums[right] == val:
                pass





if __name__ == "__main__":
    so1 = Solution()
    so1.lengthOfLongestSubstring("121")