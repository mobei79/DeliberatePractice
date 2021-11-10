# -*- coding: utf-8 -*-
"""
@Time     :2021/10/22 17:08
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
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
            nums[k+1] = temp # 因为while循环中最后都会-1


    def binary_insert_sort(self):

        for i in range(1, len(nums)):
            left = 0
            right = i-1
            mid = 0
            temp = nums[i]
            while(left <= right):
                mid = (right + left)//2
                if temp < nums[mid]:
                    right = mid - 1
                elif temp >= nums[mid]:
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
            :return:
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
    sortedBasic = SortedSolution(nums)
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

    print("\n\n\n\nLeetCode problems: ")
    ofter = [1, 12, 13, 27, 34, 38, 49, 49, 64, 65, 76, 78, 97]
    print("sorted array:\n{}".format(ofter))
    lc_problem = SortLeeCodeQuestion(nums)
    # rseK = lc_problem.quick_select_sort_pro(3)
    rseK = lc_problem.max_heap_sort(3)
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


