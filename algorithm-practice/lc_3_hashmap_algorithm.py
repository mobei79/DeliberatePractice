# -*- coding: utf-8 -*-
"""
@Time     :2021/11/9 16:20
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from typing import List
class MyHashMapQuetionSet:
    """
    哈希表 hashtable 散列表 根据关键码的值而直接进行访问的数据结构；
    查找时间复杂度为O(1)，通过索引直接查找；
    hash function：
        把名称映射到哈希表中的索引，然后通过村寻索引下标快速查找；
        一般哈希函数通过特定编码方式进行转换；
        如果hashCode值大于哈希表大小，可以对tablesize取模；
    哈希碰撞：
        拉链法：
            发生碰撞就存储在链表中，数据规模是datasize 哈希表大小为tablesize；
            既不会因空值浪费空间， 也不会因为链表太长浪费时间
        线性探索法：
            前提是tablesize > datasize，因为需要用空位来解决碰撞问题；
            如果发生碰撞，就找下一个空位存在；
    常见hash结构：
        数组；
        set集合；
            底层是哈希表
        map映射：
            key value的数据结构；key存储方式是红黑树，有一定要求；对value没有要求；
    使用场景：
        需要常数复杂度查找就需要哈希表；空间换取时间，因为需要使用额外的空间存放索引。
        判断某些值是否出现过；
    """

    """
    242. 有效的字母异位词
    给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
    注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
    
    思路和算法：
        1. 暴力法：两个for循环记录重复的次数
        2. 哈希表法：
            特点需要记录每个字母出现次数；使用数组就行，26个字母；
    """
    def isAnagram(self, s: str, t: str) -> bool:
        record = [0] * 26
        for i in range(len(s)):
            record[ord(s[i]) - ord('a')] += 1
        for j in range(len(t)):
            record[ord(t[j]) - ord('a')] -= 1
        for i in record:
            if i != 0:
                return False
        return True

    def isAnagram_dict(self, s: str, t: str) -> bool:
        from collections import defaultdict
        s_dict = defaultdict(int)
        t_dict = defaultdict(int)
        for i in s:
            s_dict[i] +=1
        for j in t:
            t_dict[j] +=1
        return s_dict == t_dict

    """
    1002. 查找共用字符
    给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按 任意顺序 返回答案。
    
    思路和解法：
        暴力法：一个个字符串搜索，复杂度O(n^m)
        技巧解法：
            如果字符c在所有字符串中出现k次及以上。那么最终结果就包含k个c。
            因此使用一个数组 minfreq[c]记录字符c在左右字符串中出现的最小次数。
            依次遍历所有字符串，如统计字符串s中，使用freq[c]统计s中每个字符出现的次数，将minfreq更新为较小的值。最终minfreq中即为结果。
            注意：
                1. 类似的字符串问题，使用数组解决就行。
                2. 逻辑中存在取最小值的操作，所以想用第一个字符串初始化freq比较好。
    """
    def commonChars(self, words: List[str]) -> List[str]:
        if not words: return []
        rst = []
        min_freq = [0]*26
        for index, c in enumerate(words[0]):
            min_freq[ord(c)-ord('a')] += 1
        for i in range(1, len(words)):
            freq = [0]*26
            for index, c in enumerate(words[i]):
                freq[ord(c)-ord('a')] +=1
            for i in range(26):
                min_freq[i] = min(min_freq[i], freq[i])
        for i in range(26):
            while min_freq[i] != 0:
                rst.append(chr(i + ord('a')))
                min_freq[i] -= 1
        return rst

    """ set
    349. 两个数组的交集
    给定两个数组，编写一个函数来计算它们的交集。
    
    思路和算法：
        不是字符问题了，数组不合适了；如果数值分散会造成空间浪费；set 其实占用空间比数组大，速度慢（因为需要hash）。
        暴力解法时间复杂度O(n^2)
        要求输出结果唯一，不考虑顺序，则可以考虑使用set
         
    """

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # sub_set = set()
        # for i in nums1:
        #     sub_set.add(i)
        # rst = set()
        # for j in nums2:
        #     if j in sub_set:
        #         rst.add(j)
        # return rst
        return list(set(nums1) & set(nums2))
        # 此处用到了set集合的用法，

    """ set
    202. 快乐数
    编写一个算法来判断一个数 n 是不是快乐数。
    「快乐数」定义为：
        对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
        然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
        如果 可以变为  1，那么这个数就是快乐数。
        如果 n 是快乐数就返回 true ；不是，则返回 false 。
    
    思路和算法：
        使用哈希，如果这个数重复出现，就return false;否则就等到sum为1为止
    注：
        1 python 平方为 **2; 除法 /返回浮点结果， // 返回征信，向下取整
    """
    def isHappy(self, n: int) -> bool:
        freq = set()
        def get_sum(n):
            sum = 0
            while n:
                sum += (n%10) ** 2
                n = n//10
            return sum
        while True:
            # if n not in freq: # 这是什么愚蠢写法
            #     sum = get_sum(n)
            #     if sum == 1:
            #         return True
            #     freq.add(sum)
            #     n = sum
            # else:
            #     return False
            n = get_sum(n)
            if n == 1:
                return True
            if n not in freq:
                freq.add(n)
            else:
                return False


    """ map
    1. 两数之和
    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
    你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
    你可以按任意顺序返回答案。
    
    题目理解：
        不重复出现，任意顺序返回，答案唯一；
        数组：有大小限制，使用元素少的情况，数值太大哈希值太大会造成时间可浪费；
        集合：元素唯一，主要用于判断是否出现过类似的问题。
    思路和解法：
        1. 暴力解法：
            遍历数组，每个值和其后面的值相加，时间复杂度O(nn)
        2. map
            遍历数组，和target求差，如果存在就返回结果。否则就将当前值和索引加入map中。
                
    """
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        rst = []
        map = dict()
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in map:
                rst.append(map[diff])
                rst.append(i)
            else:
                map[nums[i]] = i
        return rst

    """ map
    454. 四数相加 II
    给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：
    0 <= i, j, k, l < n
    nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
    题目理解：
        四个独立数组，不用考虑元素重复的情况；
    思路和算法：
        1. 定义map key存储a + b的和，value存放a+b和出现的次数；
        2. 遍历A B数组，统计map；
        3. 定义count 统计四数之和为0的次数；
        4. 遍历C D数组，找出0-（c+d）在map中出现的次数，计入count
        
    """

    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        map = dict()
        for i in nums1:
            for j in nums2:
                sum = i + j
                if sum in map:
                    map[sum] +=1
                else:
                    map[sum] = 1
        count = 0
        for i in nums3:
            for j in nums4:
                diff = 0 - i - j
                if diff in map:
                    count += map[diff]
        return count

    """ map
    383. 赎金信
    给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。
    如果可以构成，返回 true ；否则返回 false。
    (题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)
    思路和解法：
        暴力解法：两个for循环
        map解法：
            1. 使用数组统计ransom中字符出现次数；遍历magazine 如果出现key值就让value-1，最后数组空了就说明有解；
            2. 使用defaultdict
            
    """
    def canConstruct_1(self, ransomNote: str, magazine: str) -> bool:
        freq = [0]*26
        for i in ransomNote:
            freq[ord(i) - ord('a')] += 1
        for i in magazine:
            if freq[ord(i) - ord('a')] == 0:
                return False
            else:
                freq[ord(i) - ord('a')] -= 1
        return True

    def canConstruct_2(self, ransomNote: str, magazine: str) -> bool:
        from collections import defaultdict
        map = defaultdict(int)
        for x in ransomNote:
            map[x] += 1
        for x in magazine:
            if x in map:
                map[x] -= 1
        for x in map.values():
            if x > 0:
                return False
        return True

    def canConstruct_3(self, ransomNote: str, magazine: str) -> bool:
        import collections
        c1 = collections.Counter(ransomNote)
        c2 = collections.Counter(magazine)
        x = c1 - c2
        ## x 只保留大于0的符号，当c1中符号小于c2时，不会被保留。所以x留下了magazine不能表达的。
        if len(x) == 0:
            return True
        else:
            return False

    """
    15. 三数之和
    给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，
    使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
    注意：答案中不可以包含重复的三元组。
    思路和解法：
        哈希解法：
            两个for循环遍历nums，使用map存储a+b和出现的次数；然后遍历查看0-（a+b）是否有解；但是三元素不能重复的所有三元组。还需要去重就很复杂了。
        双指针法：
            1. 先将nums排序；
            2. 定义下标i， left=i+1， right在最右端；
            3. 如何移动？
                如果三数和>0，说明大了，所以right左移；
                如果三数和<0, 说明小了，left向右移动，直到相遇；
                如果等于0，说明合适了，增加i即可
            时间复杂度O(n^2)
    """

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n):
            left = i + 1
            right = n - 1
            if nums[i] > 0:
                break
            if i >=1 and nums[i] == nums[i - 1]: # 考虑排序后 相同值的情况，如果i和i-1相等，就跳过。
                continue
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum > 0:
                    right -= 1
                elif sum < 0:
                    left += 1
                else:
                    # 为什么移动？
                    #   此处i不变，看看是否还有组成求和的情况，同样需要跳过相同的值
                    ans.append([nums[i], nums[left], nums[right]])
                    while left != right and nums[left] == nums[left+1]: left+=1
                    while left != right and nums[right-1] == nums[right]: right -= 1
                    left +=1
                    right -=1
                # 因为i会增加1，如果left 和left+1相同，单纯的减少循环换次数而已。
        return ans


    """
    18. 四数之和
    给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 
        [nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：
        0 <= a, b, c, d < n
        a、b、c 和 d 互不相同
        nums[a] + nums[b] + nums[c] + nums[d] == target
        你可以按 任意顺序 返回答案 。
    思路和算法：
        三数之和使用双指针法；
        四数之和可以在外面在加一层for循环。
    """
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        rst = []
        nums.sort()
        n = len(nums)
        for i in range(n):
            if i >= 1 and nums[i] == nums[i-1]:
                continue
            for j in range(i+1,n):
                left = j+1
                right = n -1
                if j >= i+2 and nums[j]==nums[j-1]:
                    continue
                while left < right:
                    sum = nums[left] + nums[right] + nums[i] + nums[j]
                    if sum > target:
                        right -= 1
                    elif sum < target:
                        left += 1
                    else:
                        rst.append([nums[i],nums[j],nums[left],nums[right]])
                        while left < right and nums[left] == nums[left+1]: left+=1
                        while left < right and nums[right] == nums[right-1]: right-=1
                        left+=1
                        right-=1
        return rst




if __name__ == "__main__":
    exam = MyHashMapQuetionSet()
    # exam.twoSum([2,7,88], 9)
    d = dict()
    d[1] += 1
    print(d)