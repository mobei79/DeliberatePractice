# -*- coding: utf-8 -*-
"""
@Time     :2021/11/10 18:24
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
class StringAlgorithm:

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

        
    """