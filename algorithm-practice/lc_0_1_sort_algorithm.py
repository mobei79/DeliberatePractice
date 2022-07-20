# -*- coding: utf-8 -*-
"""
@Time     :2021/10/22 17:08
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
"""
leetcide： https://leetcode-cn.com/problems/sort-an-array/submissions/
1. 举一反三，梳理思想逻辑
    首先了解排序的分类：
        大分类可以分为两种：内排序和外排序：全部记录放在内存为内排序，如果需要外存为外排序、
    内排序也可分为如下几类：
        插入排序：直接插入、二分插入、希尔
        选择排序：选择排序、堆排序
        交换排序：冒泡排序、快排序
        归并排序
        基数排序
    插入排序：基本思想是每部将一个待排序的元素，按照顺序码大小插入到前面已经排好序的序列的合适位置，直到全部插入排序完成为止。
        关键问题是在前面已排好的序列中找到合适的位置？
            直接插入：按照排序从后向前和已排好序的数进行比较，找到插入位置，将该位置元素向后顺移，然后插入。
            二分插入：查找待插入位置时采用二分法，每次和中间的值进行比较，直到找到位置。
            希尔排序：即最小增量排序，等距分组；分组内采用直接排序；然后所有距离，直到距离为1。
    选择排序：基本思想是每趟从待排序的记录中选出最小的，放在已排好序列最前面。
        关键问题是在待排序序列中找到最小值的方法？
            直接选择
            堆排序：将待排序序列构建成大顶堆，然后交换顶尾的元素；重新构建，直到只剩两个；
                大顶堆展平后：a[i] >= a[2*i+1] and a[2*i+2]
                i节点的父节点时 (i-1)/2
            
    交换排序：
        冒泡排序：在待排序的序列中，自上而下的对相邻两个数依次进行比较和调整，让较大的数下沉，较小的上冒。
        快速排序：选择基准元素，一次扫描将待排序序列分为两部分，在对两侧序列进行递归处理。
    
    归并排序：归并（Merge）排序法是将两个（或两个以上）有序表合并成一个新的有序表，即把待排序序列分为若干个子序列，每个子序列是有序的。然后再把有序子序列合并为整体有序序列。
    基数排序：基本思想：将所有待比较数值（正整数）统一为同样的数位长度，数位较短的数前面补零。然后，从最低位开始，依次进行一次排序。这样从最低位排序一直到最高位排序完成以后,数列就变成一个有序序列。
2. 了解各自的时间复杂度

3. 面试常考 
    快排、堆排序、归并、桶排序
    
class SortedSolution    是基础的排序算法
    
class SortLeeCodeQuestion   LC中常见排序题目
"""
class SortedSolutionReview:
    def __init__(self, nums):
        self.nums = nums
        print("Before sort:\n{}".format(nums))
        print("right  sort:\n{}".format([1, 12, 13, 27, 34, 38, 49, 49, 64, 65, 76, 78, 97]))
    def direct_insert(self):
        """
        先检验输入；
        再i从1向后循环：
            保存比较值nums[i]
            再j从i-1向0循环（包含0 是已经排好的序列）：
                如果 temp小，则当前项后移，此时当前项还没重新赋值；
                如果 tepm大，则当前项赋值 temp
        :return:
        """
        if nums==None or len(nums)==0:
            return 0
        for i in range(1,len(nums)):
            temp = nums[i] #保存当前比较项，方便直接向后覆盖移动
            j = i - 1
            while j>=0 and temp<nums[j]:
                nums[j+1] = nums[j]
                j-=1
            nums[j+1] = temp # 因为while循环中最后都会-1

        # for i in range(1, len(nums)):
        #     temp = nums[i]  # 保存当前比较项，方便直接向后覆盖移动
        #     j = 0
        #     for j in range(i-1, -2, -1):
        #         if temp < nums[j] and j>=0:
        #             nums[j+1] = nums[j] # 如果比temp大，就后移一位，这时temp已经保存了。
        #         else:
        #             break
        #     ### 注意，java中，如果使用for且没有使用break跳出循环，最后输出的j会再进行一次++或者——
        #     nums[j+1] = temp

    def binary_insert(self):
        """
        采用二分法找插入位置
        for i in (1, len(num)):
        left = 0 right = i-1  #为已排序序列的左右边界；
        while left <= right: 等于时也需要将等于的值和temp比较
            temp > mid: left =mid+1
            temp <= mid: right = mid -1
        最后是一定是left》right，且left总为待插入的地方
        将left-(i-1)所有值向右移动一个，然后插入left=temp即可；
        :return:
        """
        if nums == None or len(nums)==0:
            return 0
        for i in range(1, len(nums)):
            temp = nums[i]
            left = 0
            right = i - 1
            mid = 0
            while left <= right:
                mid = (right + left)//2 # python中/是真正的除法有小数，//是地板除法，直接舍去小数，向下求整；java中用/
                if temp < nums[mid]:
                    right = mid - 1
                elif temp >= nums[mid]:
                    left = mid+1
            for j in range(i-1, left-1, -1):
                nums[j+1] = nums[j]
            nums[left] = temp

    def direct_select(self):
        for i in range(len(nums)):
            min = nums[i]
            min_index = i
            for j in (i+1, len(nums)):
                if nums[j] < min:
                    min = nums[j]
                    min_index = j
            if min_index != i:
                nums[min_index] = nums[i]
                nums[i] = min

    def heap_select(self):

        def build_heep(nums, last_index):
            """
            从最后一个节点开始向上调整；
            找到最后一个非叶子节点（父节点）：序列长度n，最后值索引为n-1,设最后一个父节点为i
                如果i只有左子节点：n-1 = 2i+1
                如果i有右子节点：  n-1 = 2i+2 && n-2 = 2i+1
                因为完全二叉树性质，如果只有左子节点时n为偶数，有左右时n为奇数。
                推断出，最后一个父节点的index为:n/2-1
            :param nums:
            :param last_index:
            :return:
            构建last - 0的大顶堆，last此时为nums中最后一个index
            前提最后一个父节点为(i-1)/2
            从最后一个父节点 - 0：
                k 保存当前父节点
                while（2*k+1<=lastindex）：在while中调整最后一个父节点， 条件为有左节点
                    bigindex = 2k+1 有左节点，那么最大的索引为左节点index
                    if (bigindex <= lastindex): 如果有右节点，判断两者大小，bigindex存较大子节点的索引
                        if num[bigindex] < nums[lastindex]:
                            bigindex += 1 ##即右节点的索引值
                    if nums[k] < nums[bigindex] 如果父小于子，调整
                        swap
                    else:
                        break 否则，跳出，继续调整上一个子节点
            """
            for i in range((last_index-1)//2, -1, -1): # len-1为最后一个值的index，
                k = i
                while(k*2+1 <= last_index):
                    bigindex = k*2 + 1
                    if bigindex < last_index:
                        if(nums[bigindex]<nums[bigindex+1]):
                            bigindex+=1
                    if nums[k] < nums[bigindex]:
                        nums[k], nums[bigindex] = nums[bigindex], nums[k]
                        # swap(nums[k], nums[bigindex])
                        k = bigindex
                    else:
                        break

        for i in range(len(nums)-1): # 从0 开始，0可以保证最大的调到上面来，并和最后一个交换。
            build_heep(nums, len(nums) - i - 1)
            nums[len(nums)-i-1], nums[0] = nums[0], nums[len(nums)-i-1]
            # swap(nums, 0, len(nums)-i-1)

    def heap_select_sort_2(self):
        """
        使用公共的heapify方法调整堆。
        """
        n = len(nums)
        def build_max_heap(nums, n):
            # 最后一个节点的父节点时 n//2-1
            for i in range(n//2-1, -1, -1):
                heapify(nums, n, i)
        def heapify(nums, n, i):
            """
            堆的调整过程：
            1. 保证每组子堆，满足父节点大于等于左右子节点，不满足就交换；
            2. 交换后，子节点所在的子堆可能发生变化，需要向下继续调整，知道满足条件，或者最后一个节点；
            也就是下沉;
            上浮操作指的是：构建堆的时候，在最后面插入，然后上浮处理。
            """


    def bubble_change1(self):
        for i in range(len(nums)):
            for j in range(0, len(nums) - i -1):
                if nums[j+1] < nums[j]:
                    temp = nums[j+1]
                    nums[j+1] = nums[j]
                    nums[j] = temp

    def quick_change(self):
        def quick_get_mid(nums, low, high):
            temp = nums[low]
            while low < high:
                while low < high and nums[high]>= temp:
                    high -=1
                nums[low] = nums[high]
                while low < high and nums[low] <= temp:
                    low +=1
                nums[high] = nums[low]
            nums[low] = temp
            return low

        def quick_sort(nums, low, high):
            if low < high:
                mid = quick_get_mid(nums, low, high)
                quick_sort(nums, low, mid-1)
                quick_sort(nums, mid+1, high)
        low = 0
        high = len(nums) - 1
        return nums

    def quick_change_pro1(self):
        #https://leetcode-cn.com/problems/sort-an-array/solution/kuai-pai-dai-you-hua-ban-ben-dui-pai-gui-nm0z/
        # 快排,不带优化的版本（最坏情况下会超时，所以不能通过全部测试用例）
        def quick(nums):
            if not nums:return []
            small = []; big = []
            for num in nums[1:]:
                if num < nums[0]:small.append(num)
                else:big.append(num)
            return quick(small) + [nums[0]] + quick(big)
        return quick(nums)


    def heap_select_sort(self):
        """
        增序用大顶堆；降序用小顶堆
        :return:
        """
        n = len(nums)
        def build_max_heap1(nums, n):
            # z最后一个节点索引 n-1 的父节点n//2 - 1,开始倒叙调整
            for i in range(n//2 - 1, -1, -1):
                heapify(nums, n, i)
        def heapify(nums, n, i):
            """heapify 为堆调整过程
            保证每组子堆，满足父节点大于等于左右子节点；如果不满足，则需要交换；
            交换后，发生变化的节点所在的子堆可能发生变换，需要继续调整，一直迭代下去，直到最后一个节点；
            注意：
                有一次想到，子堆发生变化，可能调换更大的值上来，导致父堆不再满足条件？
                这种理解是错的？这就是为什么要从最后一个父节点开始，如果最后一个父节点作为子节点的堆发生调整，无论调整什么值上来都比父节点小，所以父节点作为子节点所在的堆不会发生变化
                    所以只要向下递归调整即可。 heapify即为此操作；
                java注释：
                    * 如果要做到堆排序的状态，保证数组为大顶堆，或者小顶堆；在大顶堆的基础上，调换首位位置后，使用heapify进行堆调整；
                    * 此时如果即便调整，将父节点与大子节点进行调换，现在大子节点为上层的父节点，子堆发生调整，无论将那个子节点调换上来都比父节点（原来的大子节点）小，所以不会打乱父堆的顺序；
            :param nums:
            :param n: 数组的长度
            :param i: 开始调整的位置
            """


        # for i in range(len(nums) - 1):
        #     build_max_heap(nums, len(nums) - i - 1)
        #     swap


class SortedSolution:
    def __init__(self, nums):
        self.nums = nums
        print("Before sort:\n{}".format(nums) )

    def direct_insert_sort(self):
        """
        直接插入排序
            遍历将待排序数组，按照其顺序大小插入到前面已经排好序的子数组中合适的位置。
            查找插入位置的方法：
                直接插入排序：按照排序从后向前和排好的数组进行比较，找到插入的位置，将该位置上的元素依次向后顺移，然后插入目标元素
                二分法插入排序：查找待插入位置时采用二分法，每次和中间值比较
        :return:
        """
        if (nums == None or len(nums) == 0):
            return 0
        for i in range(1,len(nums)):
            temp = nums[i]
            j = 0
            # for j in range(i-1,-1, -1):
            #     if temp < nums[j]:
            #         nums[j+1] = nums[j]
            #     else:
            #         break
            # nums[j] = temp

            k = i - 1
            while k >= 0 and temp < nums[k]:
                nums[k+1] = nums[k]
                k-=1
            nums[k+1] = temp


    def binary_insert_sort(self):

        for i in range(1, len(nums)):
            left = 0
            right = i-1
            mid = 0
            temp = nums[i]
            while(left <= right):
                mid = (right + left)//2
                if temp <= nums[mid]:
                    right = mid - 1
                elif temp > nums[mid]:
                    left = mid + 1
                # else:
                #     break
            for j in range(i-1, left-1, -1):
                nums[j+1] = nums[j]
            # if left != i:
            nums[left] = temp


    def shell_insert_sort(self):
        pass

    """
    select sort
    """
    def direct_select_sort(self):
        for i in range(len(nums)):
            min = nums[i]
            min_index = i
            for j in range(i+1, len(nums)):
                if nums[j] < min:
                    min_index = j
                    min = nums[j]
            if min_index != i:
                nums[min_index] = nums[i]
                nums[i] = min

    def heap_select_sort(self):
        """
        增序，大顶堆；降序，小顶堆；
        将堆顶跟最后元素交换，剩下的元素重新构建堆；知
        :return:
        """
        def build_max_heap(nums, last_index):
            """
            找到最后一个非叶子节点：
                序列长度n = len(nums),则最后一个值的索引为n - 1; 如果最后一个父节点为i
                    如果有左子节点： n - 1 = 2*i + 1
                    如果有右子节点： n - 1 = 2*i + 2
                    java中除法是向下取整，python2中整数除法也是向下取整，python3中需要用//实现向下取整。
                    上述式子推出最后一个父节点为：i = n/2 - 1
                last_index = n - 1
                最后一个父节点 i= (last_index - 1)/2
                逆向遍历

            :param nums:
            :param last_index:
            :return:
            """
            for i in range((last_index -1)//2, -1, -1):
                k = i # 保存当前节点
                # 如果左节点存在的话，就进行调整，每次只调节当期当前节点
                while(2*k + 1 <= last_index):
                    bigger_index = 2*k + 1 # 存储大子节点的索引值
                    # 如果右节点存在，需判断是否大于左节点，改变最大子节点索引；
                    if bigger_index < last_index:
                        if nums[bigger_index] < nums[bigger_index+1]:
                            bigger_index+=1
                    # 如果当前父节点k值小于 较大的子节点值；交换两者位置；
                    if nums[k] < nums[bigger_index]:
                        swap(nums, k, bigger_index)
                        k = bigger_index # 将父节点索引改为最大子节点索引值
                        # 为什么需要这一步呢？是因为交换之后，子节点为父节点的堆结构发生改变，需要重新调整
                        # 所以在while循环中，把该父节点的子树全部调整完毕，然后在调节下一个父节点。
                    else:
                        break
        def swap(nums, i, j):
            temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp

        for i in range(len(nums) - 1):
            build_max_heap(nums, len(nums) - 1 - i)
            swap(nums, 0, len(nums) -1 -i)

    def heap_select_sort2(self):
        n = len(nums)
        def swap(nums, i, j):
            temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        def build_max_heap(nums, n):
            # 从最后一个父节点开始，逆序调整整个数组
            for i in range(n//2 - 1, -1, -1):
                heapify(nums, n, i)
        def heapify(nums, n, i):
            """
            heapify 为堆调整过程
                保证每组子堆，满足父节点大于等于左右子节点；如果不满足，则需要交换；
                交换后，发生变化的节点所在的子堆可能发生变换，需要继续调整，一直迭代下去，知道最后一个节点；
                注意：子堆发生变化，可能调换更大的值上来，导致父堆不再满足条件？这是错的， 这就是为什么要从最后一个父节点开始，如果最后一个父节点作为子节点的堆发生调整，无论调整什么值上来都比父节点小，所以父节点作为子节点所在的堆不会发生变化
                    所以只要向下递归调整即可。 heapify即为此操作；
                java注释：
                    ****如果要做到堆排序的状态，保证数组为大顶堆，或者小顶堆；在大顶堆的基础上，调换首位位置后，使用heapify进行堆调整；
                    * 此时如果即便调整，将父节点与大子节点进行调换，现在大子节点为上层的父节点，子堆发生调整，无论将那个子节点调换上来都比父节点（原来的大子节点）小，所以不会打乱父堆的顺序；
            :param nums:
            :param n: 数组的长度
            :param i: 开始调整的位置
            """
            if i > n//2 -1: # 开始调整的位置一定要小于最后一个父节点的位置；
                return
            left = 2*i + 1
            right= 2*i + 2
            max_index = i
            if left < n and nums[left] > nums[max_index]:
                max_index = left
            if right < n and nums[right] > nums[max_index]:
                max_index = right
            if max_index != i:
                swap(nums, max_index, i)
                heapify(nums, n, max_index) # 继续向下遍历

        build_max_heap(nums, n)
        for i in range(n-1, -1, -1):
            swap(nums, i, 0)
            heapify(nums, i, 0)

    def bubble_select_sort(self):
        for i in range(len(nums)):
            temp = nums[i]
            for j in range(0, len(nums) - i - 1):
                if nums[j+1] < nums[j]: # 大值 后移
                    temp = nums[j+1]
                    nums[j+1] = nums[j]
                    nums[j] = temp

    def quick_select_sort(self):
        """
        每次迭代选择一个基准，把比基准大的调到右边；循环即可找到该数在数组中的正确位置
        具体实现逻辑：
            选择low作为基准temp：
            while low < high:
                找到小于temp的最大high的位置；
                将小于temp的值赋给low； nums[low] = nums[high]
                找到大于temp的最大low的位置；
                将大于temp的值赋给high；
                    这就实现了：
            最后将temp赋值给low

        :return:
        """
        def quick_sort_middle(nums, low, high):
            temp = nums[low]
            while low < high:
                while low < high and nums[high] >= temp: # 找到小于temp的最大坐标的位置
                    high -=1
                nums[low] = nums[high] # 基准值是
                while low < high and nums[low] <= temp: # 找到大于temp的最大坐标的位置
                    low +=1
                nums[high] = nums[low]
            nums[low] = temp
            return low

        def quick_sort(nums, low, high):
            if low < high:
                mid = quick_sort_middle(nums, low, high)
                quick_sort(nums, low, mid-1)
                quick_sort(nums, mid+1, high)
        low = 0
        high = len(nums)-1
        quick_sort(nums, low, high)


class SortLeeCodeQuestion:
    def __init__(self, nums):
        self.nums = nums
        print("Before sort:\n{}".format(nums))
    """
     K个有序数组，将他们合并成一个有序数组
    """

    """
    第k大的数
    思路和算法：
        基于快排来解决，相对原数组排序，再返回倒数k个位置，平均时间复杂度为O(nlogn)，但是其实可以更快。
        只要某次划分的q为倒数第k个下标的时候，就找到了答案，不需要考虑后k个是否有序；
        改进快排算法：
            如果选取的值下标q正好是我们需要的值，直接返回即可；如果q比下标小只需要递归右子区间，否则递归左子区间，这就把递归变成了半个区间，
            提高了时间效率；快排性能和划分出的子数组长度密切相关，如果每次都是划分为1和n-1，每次递归n-1集合，这就是最坏的情况，代价O(n^2)
            引入随机化来加速这个过程，时间代价的期望是O(n)，空间复杂度O(logn),因为地柜使用栈空间的空间代价是O(logn) 「《算法导论》9.2：期望为线性的选择算法」
    注：快排是原址排序，不需要合并操作；
    补充：
        python中引用变量时要注意变量的作用域，在函数中引用不可变类型变量时，函数执行结束后不会改变全局变量的值；若想改变不可变变量的值，引用是要用global arg；
        如果是可变遍历，函数执行结束后，全局变量的值会改变；
        不可变变量：string int tuple;可变变量：dict list
    """
    def quick_select_sort_pro(self, k):
        global res
        res = 0
        def quick_sort(nums, left, right, target):
            if left > right:
                return
            l = left
            r = right
            temp = nums[left]
            while left < right:
                while left < right and nums[right] >= temp:
                    right -= 1
                nums[left] = nums[right]
                while left < right and nums[left] <= temp:
                    left += 1
                nums[right] = nums[left]
            nums[left] = temp
            if left == target:
                res = temp
                # print(res)
                return res
            else:
                # nums[l] = nums[left]
                nums[left] = temp
                if (left < target):
                    quick_sort(nums, left + 1, r, target)
                else:
                    quick_sort(nums, l, left - 1, target)


        target = len(nums) - k
        res = quick_sort(nums, 0, len(nums) - 1, target)
        return res

    """
    基于堆排序的选择方法：
        思路和算法：
            同理，只不过是使用堆排序来做。建立一个大根堆，做k-1次删除操作后，堆顶的元素就是我们要找的答案。
            时间复杂度：O(nlogn) 建堆时间代价O(n)，删除的总代价是O(klogn)，因为k远小于n，则时间复杂度O(n+klogn) = O(nlogn)
            空间复杂度 O(logn) 即递归使用栈空间的代价

    """
    def max_heap_sort(self, k):
        n = len(nums)

        def heapify(nums, n, i):
            if i > n//2 - 1:
                return
            left = 2*i + 1
            right = 2*i + 2
            max_index = i
            if left < n and nums[left] > nums[max_index]:
                max_index = left
            if right < n and nums[right] > nums[max_index]:
                max_index = right
            if max_index != i:
                nums[max_index], nums[i] = nums[i], nums[max_index]
                heapify(nums, n, max_index)

        def build_max_heap(nums, n):
            for i in range(n//2 - 1, -1 ,-1):
                heapify(nums, n, i)

        build_max_heap(nums, n)
        for i in range(n-1, n-k, -1): # 只需要移动k-1次就行，所以
            nums[i], nums[0] = nums[0], nums[i]
            heapify(nums, i, 0)
        print(nums[0])

if __name__=="__main__":
    nums = [49, 38, 65, 97, 76, 13, 27, 49, 78, 34, 12, 64, 1]
    # sortedBasic = SortedSolution(nums)
    ### 插入排序：直接插入排序，二分查找插入排序
    # sortedBasic.direct_insert_sort()
    # sortedBasic.binary_insert_sort()
    # sortedBasic.shell_insert_sort() 希尔排序

    ### 选择排序：直接选择排序，堆排序
    # sortedBasic.direct_select_sort()
    # sortedBasic.heap_select_sort()
    # sortedBasic.heap_select_sort2()
    # sortedBasic.bubble_select_sort()
    # sortedBasic.quick_select_sort()
    # sortedBasic.merge_select_sort() 归并排序
    # print("Ofter Sort:\n{}".format(nums))

    # print("\n\n\n\nLeetCode problems: ")
    # ofter = [1, 12, 13, 27, 34, 38, 49, 49, 64, 65, 76, 78, 97]
    # print("sorted array:\n{}".format(ofter))
    # lc_problem = SortLeeCodeQuestion(nums)
    # # rseK = lc_problem.quick_select_sort_pro(3)
    # rseK = lc_problem.max_heap_sort(3)
    # print("Ofter Sort:\n{}".format(nums))

    ssReview = SortedSolutionReview(nums)
    ssReview.heap_select()
    print("Ofter Sort:\n{}".format(nums))


"""
排序总结：
堆排序：
    堆排序是选择排序，主要就是构建堆结构，交换堆顶堆尾元素并重建迭代；
    构造初始堆的时间复杂度O(n),在交换重建堆的过程中，需要交换n-1次，重建时根据完全二叉树的性质，
    [log2(n-1), log2(n-2),...,1]逐渐递减，近似为nlogn，故认为是O(nlogn)的复杂度
快排：
    时间复杂度O(nlogn)
"""


