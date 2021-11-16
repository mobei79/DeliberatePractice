# -*- coding: utf-8 -*-
"""
@Time     :2021/11/10 18:24
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from typing import List

class StringAlgorithm:
    """

    """
    """
    344. 反转字符串
    编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
    不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
    思路和算法：
        不能使用额外空间；双指针法
    """
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        l = 0
        r = len(s) - 1
        while l < r:
            s[l], s[r] = s[r], s[l]
            l+=1
            r-=1
        return s

    """
    541. 反转字符串 II
    给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。
    如果剩余字符少于 k 个，则将剩余字符全部反转。
    如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
    """
    def reverseStr(self, s: str, k: int) -> str:
        s = list(s)
        def reverseString(s: List[str], l, r) -> None:
            """
            Do not return anything, modify s in-place instead.
            """
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
            return s
        n = len(s)
        l = 0
        while(l < n):
            r = l + k -1
            if r > n - 1:
                r = n - 1
            reverseString(s, l, r)
            l += 2*k
        return "".join(s)
    def reverseStr_full(self, s: str, k: int) -> str:
        def reverseString(s: List[str], l, r) -> None:
            """
            Do not return anything, modify s in-place instead.
            """
            while l < r:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
            return s
        s = list(s)
        for cur in range(0, len(s), 2*k):
            reverseString(s, cur, cur + k -1)
        return "".join(s)



class StringAlgorithm_strStr:

    """
    28. 实现 strStr()
    实现 strStr() 函数。
    给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。
    如果不存在，则返回  -1 。


    说明：

    当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

    对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。
    思路和解法：
        KMP算法：
            当字符串不匹配时，记录一部分之前匹配的文本内容，利用这些信息避免从头开始匹配。
        前缀表 next数组：prefix table
            用来回退的。记录模式串和主串不匹配时模式串从哪开始重新匹配
        最长公共前后缀：
            字符串前缀表示不包含最后一个字符的所有以第一个字符开头的连续字串
            字符串后缀表示不包含第一个字符的所有以最后一个字符结尾的连续字串
            aabaa的最长相等的前缀是aa，最长后缀也是aa，公共前后缀就是aa。匹配失败的时候是后缀的后面，所以我们只要找到公共前缀的后面重新开始匹配即可。
        如何计算前缀表：
            0 1 2 3 4 5
            a a b a a f
            长度为前1个字符的字串a,最长公共前后缀长度为0，（字符串前缀不包含最后一个字符的所有以第一个字符开头的连续字串，后缀是不包含第一个字符的所有以最后一个字符结尾的连续字串）
            长度为前2个字符的字串aa，最长相同前后缀长度为1；
            长度为前3个字符的子串aab，最长公共前后缀长度为0；
            长度为前4个字符的子串aaba，最长公共前后缀长度为1；
            长度为前4个字符的子串aabaa，最长公共前后缀长度为2；
            长度为前4个字符的子串aabaaf，最长公共前后缀长度为0；
            我们得到最长相同前后缀长度就是前缀表中的元素，下标i之前（包含i）的字符串中有多少长度相同的公共前后缀；
            前缀表：0 1 0 1 2 0
        如何利用前缀表找到字符不匹配时指针应该移动的位置。
            如果字符不匹配时，就看前缀表中前一个字符的值，然后移动到模式串的相应下标位置即可；
            因为我们要找前面的字串的公共前后缀位置，
        前缀表和next数组？
            很多KMP算法使用next数组来进行回退，在前缀表中的值统一减一，初识位置就是-1了。

        使用next数组进行匹配：
            匹配字符串s，模式串t
        时间复杂度分析：
            生成前缀表复杂度O(m),匹配的过程复杂度O(n)，总O(m+n);
            暴力解法为O(n*m)
        如何构建next数组？
            get_next(s, t)
                初始化
                    dex = -1用来标识索引值
                    next = ['' for i in range(len(needle))]
                处理前后缀不相同的情况
                    因为dex初始为-1，i就从1开始遍历，每次比较s[i]和s[dex + 1]
                    如果不同就回退，如何回退？
                        此时next[dex]存储着前dex个子串的相同前后缀长度，所以此时长度就是next[dex];
                        注意：要求dex > -1，即前面有相同的才回退，否则维持dex=-1即可
                处理前后缀相同的情况：
                    如果相同那么dex加一，向后移动就行。
        如何使用next数组做匹配？
            在s中找t是否出现？
            初始化
                初识next表 前缀起点的位置j = -1 （对应到模式串的位置就是j+1）
            不等情况:
                如果匹配字符串当前值s[i]不等于模式串前缀起始值t[j+1] (如果j=-1表示是第一个值，此时不处理)，就回退一位 j=next[j]
            相等情况：
                如果匹配字符串当前值s[i]等于模式串前缀起始值t[j+1]，就j+=1，下一个（i+1 和 t[j+1]继续比较)
            结束：
                如果j = len(needle) -1; 表示模式串都匹配完了，此时匹配到的初识位置就是i + 1 -len(needle)
    注意：数组如果要按照索引index插入数据的话，需要初始化
        next其实是前缀表
    """
    def strStr(self, haystack: str, needle: str) -> int:
        def get_next(needle):
            next = ['' for i in range(len(needle))]
            dex = -1
            next[0] = dex
            for i in range(1, len(needle)):
                # 处理不相同的情况： 不相同就需要回退
                while dex > -1 and needle[i] != needle[dex+1]:
                    dex = next[dex]
                if needle[dex+1] == needle[i]:
                    dex += 1
                next[i] = dex
            return next
        next = get_next(needle)
        j = -1 # 因为next数组记录的起始位置就是-1，沿用而已
        for i in range(len(haystack)):
            while j > -1 and haystack[i] != haystack[j+1]:
                j = next[j] #如果不同就返回上一个的位置
            if haystack[i] == haystack[j+1]:
                j+=1
            if j == len(needle) - 1: # 如果j已经是模式串最后一个了，说明匹配完了
                return (i - len(needle) + 1)

    def strStr_start1(self, haystack: str, needle: str) -> int:
        """
        前缀表不减一的写法
        :param haystack:
        :param needle:
        :return:
        """
        def get_next(next, needle):
            j = 0
            next[0] = j
            for i in range(1, len(needle)):
                while j > 0 and needle[j] != needle[i]:
                    j = next[j-1]
                if needle[j] == needle[i]:
                    j +=1
                next[i] = j
            return next
        if len(needle) == 0:
            return 0
        next = ['' for i in range(len(needle))]
        next = get_next(next, needle)
        j = 0
        for i in range(len(haystack)):
            while j > 0 and haystack[i] != needle[j]:
                j = next[j-1]
            if haystack[i] == needle[j]:
                j+=1
            if j == len(needle):
                return i + 1 - len(needle)
        return -1

    def strStr_windows(self, haystack: str, needle: str) -> int:
        """
        暴力法：
            初始化i，j开始匹配第一个字符：
            while i < n - m + 1: 判断找头结点的停止条件
                while 不匹配， i++
                匹配之后 i+1 ,j+1
                while i < n j< m 相等：继续匹配后面的：
                    i++
                    j++
                直到跳出while 即能匹配的都匹配了
                如果j=m
                    结束
                如果j！=m
                    i从前缀的下一个开始，j归零

        """
        m = len(needle)
        n = len(haystack)
        if m == 0:
            return 0
        if n < m:
            return -1
        i = j = 0
        while(i < n - m + 1):
            # 匹配首字母
            while i < n and needle[j] != haystack[i]:
                i+=1
            if i == n:
                return -1
            # 找到首字母相等的情况继续下面逻辑
            i+=1
            j+=1
            # 如果后续相等就一直继续。
            while i < n and j < m and needle[j] == needle[i]:
                i+=1
                j+=1
            # 匹配完成
            if j == m:
                return i - j
            else:
                i = i + 1 - j # i返回前缀的第一个字母开始继续匹配
                j = 0 # j归零
            # 如果后续不等了，就回到while循环。
        return -1

    """
    459. 重复的子字符串
    给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。
    给定的字符串只含有小写英文字母，并且长度不超过10000。
    思路和算法：
        KMP算法中，next数组中 记录了最长想通过前后缀表。最长相等前后缀长度为next[len-1] + 1;
            len - (next(len-1) + 1) 表示第一个周期的长度，如果这个周期可以被整除，就能循环。
            如果len % (len - (next[len-1] +1)) == 0 最长前后缀能被len整出。说明存在重复串
        注意：next[-1] 即最后一个为-1时
    """
    def repeatedSubstringPattern(self, s: str) -> bool:
        def get_next(next, s):
            j = -1
            next[0] = j
            for i in range(1, len(s)):
                while j >=0  and s[i] != s[j + 1]:
                    j = next[j]
                if s[i] == s[j + 1]:
                    j += 1
                next[i] = j
            return next

        n = len(s)
        if n == 0:
            return False
        next = ['' for i in range(n)]
        next = get_next(next, s)
        print(next)
        if next[-1] != -1 and n % (n - (next[n - 1] + 1)) == 0:
            return True
        return False


if __name__ == "__main__":
    exam = StringAlgorithm_strStr()
    rst = exam.repeatedSubstringPattern("abac")
    print(rst)