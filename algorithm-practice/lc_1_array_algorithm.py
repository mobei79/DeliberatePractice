# -*- coding: utf-8 -*-
"""
@Time     :2021/10/18 19:32
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

from typing import List  #  类型提示支持
"""
数组是存储在连续内存空间上相同类型数据的集合，可以通过”下标索引“进行访问。
"""


class Solution:
    """
    二分查找
    704. 二分查找
    给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，
    如果目标值存在返回下标，否则返回 -1。
    思路解法：
        二分查找适用场景：有序的数组、无重复（有重复找到的下标可能不唯一）、查找应用；
    """
    def search_zbyk(self, nums: List[int], target: int) -> int:

        # 左闭右开
        left = 0
        right = len(nums) # 左闭右开所以不用减1
        while(left < right):
            mid = (left + right)//2
            if target < nums[mid]:
                right = mid # 右开 不包含
            elif target < nums[mid]:
                left = mid+1 # 左闭 所以加一
            else:
                return mid
        return -1

    def search_zbyb(self, nums, target):
        """
        左闭右闭写法 target就是在一个左闭右闭的区间中
        :param nums:
        :return:
        """
        left = 0
        right = len(nums) - 1 # 因为左闭右闭所以减一
        while (left <= right): # 因为left=right有意义所以有等号，即【1,1】表示1，有意义
            mid = (right + left) // 2
            if(nums[mid] > target):
                right = mid - 1 # 为什么减1。是因为区间左闭右闭，小于mid之后就没必要在比较了，故减一
            elif (nums[mid] < target) :
                left = mid + 1
            else:
                return mid
        return -1

    """
    35. 搜索插入位置
    给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，
    返回它将会被按顺序插入的位置。
    请必须使用时间复杂度为 O(log n) 的算法。
    思路解法：
        暴力解法时间复杂度O(n)
        二分法时间复杂度O(logn)
    """
    def searchInsert_violence(self, nums: List[int], target: int) -> int:
        # 时间复杂度O(n), 空间复杂度
        size = len(nums)
        for i in range(size):
            if nums[i] >= target:
                return i
        return size
    def searchInsert_search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) # d
        while left < right:
            mid = (left+right)//2
            if nums[mid] > target:
                right = mid # target在左区间
            elif nums[mid] < target:
                left = mid + 1  #target在右区间
            else:   # 如果存在该值，返回坐标
                return mid
        """
        需要处理序列中不存在目标值的情况
        如果[0,0)区间
        目标值在数组中，返回mid
        目标值插入数组中的位置，如果在区间中，返回right即可
        目标值不在区间中，则最后区间会收缩值[len,len),返回right即可
        """
        return right + 1    # 如果不存在该值，返回

    """
    367. 有效的完全平方数
    给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则返回 false 。
    进阶：不要 使用任何内置的库函数，如  sqrt 。
    思路解法：
        二分查找
            如果num<2 返回true；
            在边界[2，num/2]
            令mid=(left+right)/2
                如果mid^2 == num 为完全平方数，返回true
                如果mid^2 < num   left = mid + 1
                如果mid^2 > num   right = mid
            循环没找到。返回flase
        
        牛顿迭代法：
            找出 f(x) = x^2 - num =0 的根
            牛顿迭代法的思想是从一个初始近似值开始，然后作一系列改进的逼近根的过程。
            
        
    """
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        left, right = 2, num//2
        while left <= right:
            mid = (left + right)//2
            squ = mid * mid
            if squ < num:
                left = mid + 1
            elif mid*mid > num:
                right = mid - 1
            else:
                return True
        return False

    def isPerfectSquare_newton(self, num: int) -> bool:
        # https://leetcode-cn.com/problems/valid-perfect-square/solution/you-xiao-de-wan-quan-ping-fang-shu-by-leetcode/
        """
        取x作为初始近似值，然后在（x,f(x)）处做切线，与x轴相较经过x_k+1,
        通过切线斜率科协等价公式： x_k+1 = x - f(x)/f'(x)，
        带入公式
            f(x)=x^2 - num
            f'(x) = 2x
        得到
            x_k+1 = 1/2(x+num/x)
        算法：
            取num/2作为初始近似值
            当x*x > num,用牛顿法计算下一个近似值     x_k+1 = 1/2(x+num/x)
            返回 x*x = num
        时间复杂度：O(logN)
        :param num:
        :return:
        """
        if num < 2:
            return True
        x = num//2
        while x*x > num:
            x = (x + num//x) //2
        return x*x == num

    """
    27. 移除元素
    给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
    元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
    示例 1: 给定 nums = [3,2,2,3], val = 3, 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。 你不需要考虑数组中超出新长度后面的元素。
    示例 2: 给定 nums = [0,1,2,2,3,0,4,2], val = 2, 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
    你不需要考虑数组中超出新长度后面的元素。
    思路解法：
        暴力解法：
            两个for循环，外层遍历数组，如果找到目标值，在内层循环更新【从i+1到size，依次往前移动一位即可，可以覆盖】
        指针法： O(n)  O(1)
            
            初始化快慢指针 r,l = 0
            for循环：
                如果等与目标值 l不变，r+1
                如果不等于目标值：
                    如果l != r: num[l] = num[r]
                    l++
                    r++
    双指针法：在数组和链表的操作中非常常见，很多数组，链表，字符串都会用到双指针           
    """

    def removeElement_violence(self, nums: List[int], val: int) -> int:
        rst = 0
        size = len(nums)
        for i in range(size):
            while nums[i] == val:
                for j in range(i+1, size):
                    nums[j-1] = nums[j]
                    size-=1
            rst +=1
        return rst

    def removeElement(self, nums: List[int], val: int) -> int:
        # rst = 0
        # left, right = 0,0
        # for i in range(len(nums)):
        #     if nums[right] == val:
        #         right +=1
        #     else:
        #         if left != right:
        #             nums[left] = nums[right]
        #         left +=1
        #         right +=1
        #         rst += 1
        # return rst
        rst = 0
        left, right = 0, 0
        while right < len(nums):
            if nums[right] != val:
                nums[left] = nums[right]
                left += 1
            right+=1
            # 当快指针遇到需要要删除的元素是；left指针停止移动，right继续移动
            # 如果不是要删除的元素，就将值赋给left指针，然后都移动
            # 如果快指针 right>len 停止
        return left

    """
    26. 删除有序数组中的重复项
    给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
    不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
    思路解法：
        双指针解法：
            定义左右指针，右指针=左指针+1
            如果左右指针相等，右指针+1，左指针不变;
            如果左右指针不相等，num[l+1] = num[r] l++,r++
        注：右指针至少比左指针大1；右指针到头就停止，数组长度为左指针索引+1
    """
    def removeDuplicates(self, nums: List[int]) -> int:
        # if nums == None or len(nums)<=1:
        #     return len(nums)
        # rst = 0
        # l,r = 0,1
        # while r < len(nums):
        #     if nums[r] != nums[l]:
        #         nums[l+1] = nums[r]
        #         l+=1
        #     r +=1
        # return l+1
        left=0
        for right in range(1,len(nums)):
            if nums[right] != nums[left]:
                nums[left+1]=nums[right]
                left+=1
        return left+1

    """
    283. 移动零
    给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
    """
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        for right in range(0, len(nums)):
            if nums[right] != 0:
                temp = nums[left]
                nums[left] = nums[right]
                nums[right] = temp
                left+=1
            right+=1
        return None

    """
    844. 比较含退格的字符串
    给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，请你判断二者是否相等。# 代表退格字符。
    如果相等，返回 true ；否则，返回 false 。
    注意：如果对空文本输入退格字符，文本继续为空。
    思路和解法：
        1. 重构字符串
            即将字符串中退格符和对应的元素去除，还原成一般形式；
            需要遍历两个字符串；
            时间复杂度O(M+N);空间复杂度O(M+N)
        2. 双指针
            一个字符是否会删掉，只取决于该字符后面的退格符，所以逆序
            逆序一个一个的对比，主要是代码实现时的写法问题；
            主循环的判断使用while i >=0 or j >=0
                内循环使用while i >=0, 
                    如果是#，就消除一个，然后再选择另一个
                    如果不是#，标志位设置为1，
                判断两个数组是否都取到值了
                判断取到的值是否相等
    """
    def backspaceCompare_rebuild(self, s: str, t: str) -> bool:
        def build(s:str)->str:
            rst = list()
            for ch in s:
                if ch!='#':
                    rst.append(ch)
                elif ch=='#' and rst:
                    rst.pop()
            return "".join(rst)
        return build(s)==build(t)

    def backspaceCompare_doublePoint(self, s: str, t: str) -> bool:

        i, j = len(s)-1, len(t)-1
        skipi, skipj =0,0
        while(i >=0 or j >= 0):
            while i >= 0:
                if s[i] == '#':
                    skipi +=1
                elif skipi:
                    skipi -=1
                else:
                    break
            while j >= 0:
                if s[j] == '#':
                    skipj +=1
                elif skipj:
                    skipj -=1
                else:
                    break
            if i>=0 and j>=0:
                if s[i] != t[j]:
                    return False
            elif i>=0 or j >= 0:
                return False
            i-=1
            j-=1

    """
    977. 有序数组的平方
    给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，
    要求也按 非递减顺序 排序。
    思路解法：
       分析：
            非递减顺序：即增序但是可能存在重复
            因为整数的平方仍然有序，所以主要考虑负数的情况；
            如果只有正数或者只有负数，则按照顺序即可；如果同时存在正负数，则需要按照绝对值大小进行比较
        暴力解法：
            每个数去平方，然后排序；
        左右指针法：
            比较绝对值，然后left++或者right--
            
    """
    def sortedSquares_violence(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            nums[i] = nums[i]*nums[i]
        nums.sort()
        return nums
    def sortedSquares(self, nums: List[int]) -> List[int]:
       left, right = 0,len(nums)-1
       rst = []
       while left < right:
           if abs(nums[left]) > abs(nums[right]):
               rst.append(nums[left]*nums[left])
               left+=1
           else:
               rst.append(nums[right]*nums[right])
               right-=1
       return rst

    """
    209. 长度最小的子数组
    给定一个含有 n 个正整数的数组和一个正整数 target 。
    找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，
    并返回其长度。如果不存在符合条件的子数组，返回 0 。
    思路解法：
        暴力解法：
            两个for循环，外层遍历数组，内层遍历求和，找到最符合的子序列，时间复杂度O(N^2)
        滑动窗口：
            不断调节子序列的起始位置和终止位置，这样只用遍历一次即可；时间复杂度O(N)
            双指针其实可以理解为双指针法的一个变种，只不过不是首尾而是窗口滑动的形式。
            注意三点;
                窗口内是什么，要求什么；        -- 窗口数和>=s
                如何移动起始位置和终止位置；     -- 如果窗口值>=s，起始位置就移动，否则就往后移动结束为止，
            编程注意点：
                初始化起始节点为0，
                然后遍历数组作为终止节点。
                    sum+=nums(j)
                    while sum >= target         # 使用while是因为加上的数可能很大，删除掉起始节点还是大于target
                        更新最短长度
                        减去起始值
                        左节点加一
                    
    """
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        rst = float("inf") # 定义一个无限大的数
        sum = 0
        sublen = 0
        for j in range(len(nums)):
            sum += nums[j]
            while sum >= target:
                rst = min(rst,j-left+1)
                sum -= nums[left]
                left+=1
        return 0 if rst==float("inf") else rst



if __name__=="__main__":
    solution = Solution()
    rst = solution.search_zbyk([-1,0,3,5,9,12], 13)
    # rst = solution.sortedSquares_violence([0,1,2,2,3,0,4,2])
    print(rst)