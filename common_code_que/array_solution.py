# -*- coding: utf-8 -*-
"""
@Time     :2021/6/11 16:19
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

class ArraySolution:

    """
    :题目 1
    在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，
    也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    :示例
    输入：
    [2, 3, 1, 0, 2, 5, 3]
    输出：2 或 3
    :param nums:
    :return:
    """


    def find_repeat_number_dict(self, array_list: list) -> int:

        # 方法1 哈希表 时间复杂度O(n）空间复杂度O(n)
        dict_times = dict()
        for item in array_list:
            if item not in dict_times:
                dict_times[item] = 1
            else:
                return item

    def find_repeat_number_set(self, array_list: list) -> int:
        # 方法2 set 时间复杂度O(n）空间复杂度O(n)
        set_times = set()
        for item in array_list:
            if item not in set_times:
                set_times.add(item)
            else:
                return item

    def find_repeat_number_rm(self, array_list: list) -> int:
        # 方法3 根据 nums 里的所有数字都在 0～n-1 的范围内 时间复杂度O(n) 空间复杂度O(1)
        # 上面的方法是采用python的数据结构特性来解决问题，这里是采用算法是想解决问题。
        num_max = max(array_list)
        for i in range(0, num_max+1):
            if i in array_list:
                array_list.remove(i)
        return array_list[0]

    def find_repeat_number_alog(self, array_list: list) -> int:
        # 方法 4 空间O(1) 时间O(1)
        # '''
        # 这也属于一种字符串 查重算法，适用于从(0, n-1)中取数得到长为n的数组；
        # 遍历enumerate，如果index=value相同标为-1；
        # 如果index != value, 将value作为下标的值标为-1表示该数已经出现过一次了，
        #     然后以对应的value作为下标迭代；
        # 如果任意一次value = -1 说明出现过一次了。
        # '''
        repeated_item = []
        for index, value in enumerate(array_list):
            if index == value:
                array_list[index] = -1
                continue
            if value == -1:
                continue
            while value > -1:
                item = array_list[value]
                if item == -1:
                    return value
                else:
                    array_list[value] = -1
                    value = item
        return repeated_item


    def find_number_in_2Darray(self, matrix: list, target: int) -> bool:
        pass


if __name__ == "__main__":
    instance_a_s = ArraySolution()
    test_list = [2, 1, 3, 4, 3, 4, 6, 5, 5, 7]
    rst_alog = instance_a_s.find_repeat_number_alog(test_list)
    print(rst_alog)
