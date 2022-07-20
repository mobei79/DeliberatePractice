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

    """
    151. 颠倒字符串中的单词
    给你一个字符串 s ，颠倒字符串中 单词 的顺序。
    单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
    返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
    注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
    """
    def reverseWords(self, s: str):
        return [ x.split() for x in s.split(" ")]



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

class LC18RomanToInt:

    SYMBOL_VALUE = {
        "I":1,
        "V":5,
        "X":10,
        "L":50,
        "C":100,
        "D":500,
        "M":1000
    }

    def romanToInt(self, s:str) -> int:
        """
        时间复杂度 O(n) 空间复杂度O（1）
        :param s:
        :return:
        """
        ans = 0
        n = len(s)
        for i, st in enumerate(s):
            value = self.SYMBOL_VALUE.get(st)
            if i < n-1 and value < self.SYMBOL_VALUE[s[i+1]]:
                ans -= value
            else:
                ans += value
        return ans

"""
剑指 Offer 67. 把字符串转换成整数
    写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
    
    首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
    当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
    该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
    注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
    在任何情况下，若函数不能进行有效的转换时，请返回 0。
    说明：
    假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    "空格问题、非数字字符问题，数字范围，正负号"

题解：
    主要考虑越界问题：
        1 首部符号：”空格“删除、”符号位“+—或者空格新建一个变量保存符号，"非数字字符"：直接返回0
        2 数字字符：
            数字字符转换：将这个字符的ASCII码减去0的ASCII码即可
            数字拼接：从左向右遍历，设置当前字符c，数组对应x，结果设置为res，拼接公式为res = 10*res + x;
            数字越界处理：每轮拼接之后，判断拼接值是否超过[-2^31, 2^32-1]。因为进位需要*10，所以设置边界band = 2^31-1//10
                如果res > band ;拼接后越界，直接返回极值
                如果res= band ;x>7 拼接后也是越界
"""
def LC_OFFER67StrToInt(str:str)-> int:
    """
    :param s:
    :return:
    """
    # str = str.strip() # 去除首尾空格
    # if not str: return 0 # 字符串为空
    # res, i, sign = 0, 1, 1 # i用于标记其实位置，默认是+-，如果是负号sign=-1,如果是正号sign=1默认；如果没有正号，说明从第二个字符开始
    # int_max, int_min, band = 2**31-1, -2**31, 2**31//10
    # if str[0] == "-": sign = -1 # 判断符号
    # elif str[0] != "+": i = 0
    # for c in str[i:]:
    #     if not '0' <=c<= '9': break
    #     if res> band or res==band and c>'7': return int_max if sign==1 else int_min # 数字越界处理
    #     res = res*10 + ord(c) - ord('0') # 字符拼接
    # return res*sign

    # 设置所有需要的初识变量
    res, i, sign, length = 0, 0, 1, len(str)
    max, min, band = 2**31-1, 2**31, 2**31//10
    # 判断空值
    if not str: return 0
    # 过滤前面的空值
    while str[i] == " ":
        i+=1
    # 判断符号位
    if str[i] == "-":
        sign = -1  # 默认是正，所以只处理负号
    if str[i] in "+-":
        i+=1     # 如果有正负号，向后移动一位
    for c in str[i:]:
        if not '0' <=c<='9': break # 如果出现非数字字符，直接返回
        # 先判断是否越界
        if res>band or res==band and 'c' > '7':
            res = max if sign == 1 else min
        res = res*10 + ord(c) - ord('0')
    return res*sign

"""
剑指 Offer 48. 最长不含重复字符的子字符串
    请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
    示例 1:
    输入: "abcabcbb"
    输出: 3 
    解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
    示例 2:
    输入: "bbbbb"
    输出: 1
    解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
    示例 3:
    输入: "pwwkew"
    输出: 3
    解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
         请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
    提示：
    s.length <= 40000
题解：
我的思路：（起始就是双指针，线性遍历）从前向后遍历，如果出现重复字符，保存当前长度。从重复字符的起始位置继续向后遍历
LC思路： 动态规划/双指针+哈希表：
    暴力解法：长度为N的字符串共有（1+N）*N/2个子字符串，复杂度O(n^2)。判断长度为N的字符串是否含有重复字符复杂度为O(N),本体暴力求解复杂度O(N^3)
    BP数组思想：
        状态定义：假设动态规划列表dp。dp[j]表示义字符s[j]结尾的最长不重复字串长度；
        转义方程：固定右边界j，设字符s[j]左边距离最近的相同字符为s[i]即s[j]=s[i]
            当i<0，即s[j]左边无相同字符。dp[j] = dp[j-1]+1
            当dp[j-1] < j-i,说明s[i]在子字符串dp[j-1]区域之外，dp[j]=dp[j-1]+1
            当dp[j-1]>=j-i，说明字符s[i]在子字符串dp[j-1]区间之中，则 dp[j]的左边界由s[i]决定，dp[j]=j-i;
            
            当i<0时，dp[j-1]<=j恒成立，因此总的转义方程如下：
            dp[j]=dp[j-1]+1 ；dp[j-1]<j-i
            dp[j]= j-i       dp[j-1]>=j-i
        DP返回值：max(dp)
        复杂度分析：
            【此方法节省dp列表使用的O(N)空间】由于只取dp数组最大值，因此借助变量temp存储dp[j],变量res存储每轮更新最大值即可；
    dp数组+哈希表： 哈希表用于计算索引i的位置
        哈希表统计：遍历s时，dic统计各个字符最后一次出现的位置。
        左边界i获取方式：遍历s[j]时，通过访问哈希表dic[s[j]]获取最近的相同字符索引i。
        时间复杂度：O(N) 空间复杂度:O(1)【字符的ascii码范围0-127，哈希表dic最多使用O(128)=O(1)的额外空间】
    dp数组+线性遍历：
        左边界获取方法：遍历s[j]时，初始化索引i=j-1,向左遍历搜索第一个满足s[i]=s[j]的字符即可
        时间复杂度：O(N^2)【动态规划遍历计算dp列表需要O(N),每轮搜索i的位置需要遍历j个字符，占用O(N)】空间复杂度O(1)
    双指针+哈希表：
        哈希dic：指针j遍历字符s，哈希表统计字符s[j]最后一次出现的索引
        更新左指针i：根据上轮左指针i和dic[s[j]]，每轮更新左边界，确保区间[i+1,j]内部无重复字符
        更新结果res:取上轮res和本轮指针区间宽度，中最大值。
        时间复杂度O(N) 空间复杂度O(1),哈希表最多使用常数级额外空间。    
作者：jyd
链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/solution/mian-shi-ti-48-zui-chang-bu-han-zhong-fu-zi-fu-d-9/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
        
"""
def LCOFFER48LengthOfLongestSubstring_myself(str):
    if not str:return 0
    max_len = 0
    start = 0
    sub = []
    for c in str:
        if c in sub:
            index = sub.index(c)
            if index == len(sub)-1:
                sub = []
            else:
                sub = sub[index+1:]
            start += index + 1
        sub.append(c)
        local_len =len(sub)
        if local_len > max_len:
            max_len = len(sub)
    return max_len
def LCOFFER48LengthOfLongestSubstring_dp_dict(s):
    res, dptmp, i =0,0,0 # dp存储每轮迭代的dp值，即dp[j-1],节省空间
    dic = {}
    for j in range(len(s)):
        i = dic.get(s[j],-1) # 获取最相似的索引位置
        dic[s[j]] = i # 更新dic,s[j]字符最近相似的索引位置
        dptmp = dptmp + 1 if dptmp<j-1 else j-1
        res = max(res, dptmp)
    return res

def LCOFFER48LengthOfLongestSubstring_dp_line(s):
    res, tmp, i = 0,0,0
    for j in range(len(s)):
        i = j-1
        while i >=0 and s[i]!=s[j]:
            i-=1
        tmp = tmp+1 if tmp<j-i else j-i
        res = max(res, tmp)
def LCOFFER48LengthOfLongestSubstring_double_dict(s):
    res, i = 0, -1
    dic = {}
    for j in range(len(s)):
        if s[j] in dic: # 检查是否有重复值，更新左指针i，
            i = max(dic[s[j]], i)
        dic[s[j]] = j # 更新哈希表索引值
        res = max(res, j-i)
    return res


"""440. 字典序的第K小数字
给定整数 n 和 k，返回  [1, n] 中字典序第 k 小的数字。
    示例 1:
    输入: n = 13, k = 2
    输出: 10
    解释: 字典序的排列是 [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]，所以第二小的数字是 10。
    示例 2:
    输入: n = 1, k = 1
    输出: 1
    提示:1 <= k <= n <= 109

思考：
    什么是字典序？如何定位这个数？
    字典序：就是根据数字的前缀进行排序。10<9的前缀是1；112<12前缀11小于12
        画图后，字典序其实就是10叉树的先序遍历，1,10,100,101,102... 11,110,111,112...
    找到排位k的数？
        【找排位k的数，最好的方法是按照前缀的顺序，如果k不在当前前缀下，直接向后寻找。】
        需要搞清楚三个事情：
        如何确定前缀下所有子节点的个数？
            用下一个前缀的起点，减去当前前缀的起点，就是当前前缀下所有节点数目总和。
            假设prefix是前缀，n是上界：
        如果第k个数在当前前缀下，怎么继续往下面的子节点找？    
思路和解法：
    

"""
def findKthNumber(n: int, k: int) -> int:
    def getCount(prefix, n):  # 计算当前前缀下节点个数
        cur = prefix  # 当前前缀，
        next = prefix + 1  # 下一个前缀
        count = 0
        while cur <= n:  #
            # count = next - cur # 下一个前缀起点，减去当前前缀起点。
            """
            #next大于上界的情况，当前前缀就不是满树，所以next -cur不对
            n+1 是因为当前子树包含根节点；如果n=13，则1为前缀的根为10，13-10=3,有四个叶子节点。
            """
            count += min(n + 1, next) - cur
            cur *= 10  # 前缀下的数值
            next *= 10
            """
            如果当前前缀1，next=2,分别变成10，和20；1为前缀的节点增加10个，十叉树增加一层
            如果现在cur=10，就变成了100，1为前缀的10叉树又增加了1层。
            """
    p = 1 # 指针，指向当前位置的数值【节点的值就是当前数值】，当p=k时，即找到第k个数
    prefix = 1 # 初始前缀
    while(p < k):
        count = getCount(prefix, n)
        if p + count > k:
            prefix *=10 # 说明在当前前缀下，当前前缀变成第一个叶子节点。
            p+=1 # 把指针指向第一个叶子的位置，如prefix=11, 变成110，p加1【这是按照字典序加1】从11指向110
        elif p+count <= k:
            prefix += 1# 说明不在当前前缀下，遍历后一个前缀
            p += count  # 将指针指向下一个前缀出处
    return prefix
def findKthNumber_pro(n: int, k: int) -> int:
    """
    本质上是一个10叉树的先序遍历问题，找到按照先序遍历的第k和节点。
    为什么是先序： 字典序的性质决定的，相同位数的数字，在10叉树的同一层。
    从cur=1开始遍历，先计算以cur为根且子节点值<n的节点个数nodes
        如果nodes<=k，说明cur开头的子树中合格节点数不够，cur同层右移。cur++
        如果nodes>k, 说明cur开头的子树中节点足够，cur应该往下走，cur*=10 [再在下一层重复]
    """
    def getNodes(cur, k):
        # 计算【1，n】范围内，cur为根的节点个数。（包含cur）
        next = cur + 1
        nodes = 0
        while cur <= n: # 如果cur小于最大数就可以进入循环
            # 如果n不在cur层，当前层有效节点数为next-cur 其实就是1；下面不管多少层
            # 如果n在cur层，当前层有效节点数为n-cur+1
            nodes += min(n-cur+1, next-cur)
            cur *=10
            next *=10
        return nodes


    cur = 1 # 开始遍历的位置 以 1开头；
    k -= 1 # 因为从1开始，所以k值减一。  ？？？ 不应该刚好是1么？
    while(k>0):
        nodes = getNodes(cur, n)
        if nodes <= k:
            cur+=1
            k-=nodes
        else:
            cur *= 10
            k -= 1
    return cur

class Solution:

    """
    剑指 Offer II 018. 有效的回文
    给定一个字符串 s ，验证 s 是否是 回文串 ，只考虑字母和数字字符，可以忽略字母的大小写。
    本题中，将空字符串定义为有效的 回文串 。
    示例 1:
    输入: s = "A man, a plan, a canal: Panama"
    输出: true
    解释："amanaplanacanalpanama" 是回文串
    示例 2:
    输入: s = "race a car"
    输出: false
    解释："raceacar" 不是回文串
    提示：
    1 <= s.length <= 2 * 105
    字符串 s 由 ASCII 字符组成
    解法和思路：
        双指针：
    """
    def isPalindrome(self, s: str) -> bool:
        # def is_char(c):
        #     # return 'a'<=c<='z' or 'A'<=c<='Z' or '0'<=c<='9'
        #     return str.isalnum(c)
        # def is_equals(a,b):
        #     a = a.lower()
        #     b = b.lower()
        #     return a == b
        left = 0
        right = len(s) - 1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if left < right and not s[left].lower() == s[right].lower():
                return False
            left += 1
            right -= 1
        return True
    """409. 最长回文串
    给定一个包含大写字母和小写字母的字符串 s ，返回 通过这些字母构造成的 最长的回文串 。
    在构造过程中，请注意 区分大小写 。比如 "Aa" 不能当做一个回文字符串。   
    示例 1:
    输入:s = "abccccdd"
    输出:7
    解释:
    我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
    示例 2:
    
    输入:s = "a"
    输入:1
    示例 3:
    
    输入:s = "bb"
    输入: 2
     
    
    提示:
    
    1 <= s.length <= 2000
    s 只能由小写和/或大写英文字母组成
    思路和解法：
        如果字符有两个就可以组成回文，将问题转化为查找可以成对的字母;
        解法1：新建list存储遍历过得字符，遇到相同的就删除，并+2； 这样占用大量额外空间；
        解法2：定义128维度的数组，存储字符出现的次数；因为ascii码为128，
        解法3：使用hashmap dict 存储出现次数，因为只有大小写字母，占用空间为52
    """
    def longestPalindrome(self, s: str) -> int:
        sub = []
        rst = 0
        for i in s:
            if i in sub:
                sub.remove(i)
                rst+=2
            else:
                sub.append(i)
        if sub: rst+=1
        return rst

    def longestPalindrome_pro(self, s: str) -> int:
        from collections import Counter
        dict = Counter(s)
        is_add = False
        rst = 0
        for v in dict.values():
            rst += v
            if v%2 == 1:
                is_add = True
                rst-=1
        return rst+1 if is_add else rst

    def longestPalindrome_dict(self, s: str) -> int:
        dic = {}
        rst = 0
        for i in s:
            if i in dic:
                if dic[i] >0:
                    dic[i] -=1
                    rst+=2
                else:
                    dic[i] +=1
            else:
                dic[i] +=1
        for i in dic.values():
            if i > 0:
                rst+=1
            break
        return rst

    def longestPalindrome_dict_pro(self, s: str) -> int:
        dic = {}
        rst = 0
        for i in s:
            if i in dic:
                dic[i] +=1
            else:
                dic[i] = 1
        is_sigl = False
        for j in dic.values():
            rst += j
            if j%2 == 1:
                is_sigl =True
                rst -=1
        return rst+1 if is_sigl else rst

    def longestPalindrome_list(self, s: str) -> int:
        arr = [0]*128
        rst = 0
        count_sigle = 0
        for i in s:
            arr[ord(i)] +=1
        for i in arr:
            count_sigle += i%2
        return len(s) if count_sigle == 0 else len(s) - count_sigle + 1

    """5. 最长回文子串
        给你一个字符串 s，找到 s 中最长的回文子串。
        示例 1：
        输入：s = "babad"
        输出："bab"
        解释："aba" 同样是符合题意的答案。
        示例 2：
        输入：s = "cbbd"
        输出："bb"
        提示：
        1 <= s.length <= 1000
        s 仅由数字和英文字母组成
    思路和解法：
        中心扩散法：时间复杂度O(N^2) 长度为1和2的中心，分别有n和n-1个，每个回文中心最多扩展n次
        动态规划：
            dp[i, j] 子字符串s[i,j]组成的串是否是回文：
            转移矩阵：
                dp[i,j]= dp[i+1,j-1]  s[i] = s[j]
                
                  
    """
    def longestPalindrome_mid(self, s: str) -> str:
        def expandArountCenter(s, left, right): # 找到从left right向两侧扩展的最大回文串
            while left>0 and right<len(s) and s[left]==s[right]:
                left-=1
                right+=1
            return left+1, right-1
        if s==None or len(s)==0:return ""
        strlen = len(s)
        start = 0
        end = 0
        for i in range(len(s)):
            left1, right1 = expandArountCenter(s, i, i)
            left2, right2 = expandArountCenter(s, i, i+1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start:end+1]

    def longestPalindrome_dp(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s

        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True

        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]

                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]




if __name__ == "__main__":
    exam = StringAlgorithm_strStr()
    rst = exam.repeatedSubstringPattern("abac")
    print(rst)
