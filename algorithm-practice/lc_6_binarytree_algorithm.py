# -*- coding: utf-8 -*-
"""
@Time     :2021/11/12 10:01
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""
from collections import deque
import typing
from typing import List
class BinaryTree:
    """
    知识点：
        递归三部曲：返回值和参数；终止条件；单层逻辑；
        递归对应的就是迭代法，优先使用递归；
        二叉树种类、便利方式、定义方式：
            满二叉树
            完全二叉树：除了最底层没填满，且是几种在左边；
            二叉搜索树：
                有序树；如果左子树不为空，左子树上所有节点均小于根节点，右子树均大于根节点，左右子树为二叉排序树；
            平衡二叉搜索树：
                左右子树的高度差绝对值不超过1；
            c++中set map喝多都是平衡二叉树；增删时间复杂度O(logn)；但是有的map和set底层是哈希表；
        存储方式：
            链式存储就是使用指针；一般用链式存储
            顺序存储就是使用数组，根节点下标i左右子孩2i+1和2i+2
        二叉树遍历：
            深度优先：前中后序
            广度优先：通过队列模拟
        二叉树属性类：
            是否对称；1后序，比较根节点的左右节点是不是相互翻转；2
            树的深度；
            节点数目；
            所有路径： 递归迭代回溯
            是否平衡
            求左叶子之和
            求最小角的值
            求路径总和
        二叉树修改改造
            翻转二叉树
            构造二叉树
            构造最大二叉树
            合并另个二叉树
        二叉搜索树
    """
    pass



class TreeNode:
    def __init__(self, val):
        left = None
        right = None
        self.val= val

    """递归法"""
    def pre_order_traversal(self, root):
        """
        递归调用是，要传入rst，这样rst才会在每个递归调用中修改；注意如果return值，需要在递归函数入口赋值；
        1. 确定递归函数 参数（那些参数需要递归处理）和返回值（明确递归返回值进而确定递归函数返回类型）：
        2. 确定终止条件：注意如果终止条件return值了一定要注意
        3. 确定单层逻辑。
        :param root:
        :return:
        """
        def traversal(cur, rst): # 传入rst才会在递归中改变值
            if cur == None:
                return
            rst.append(cur.val)
            traversal(cur.left, rst)
            traversal(cur.rigth, rst)

        rst = []
        traversal(root, rst)

    def in_order_traversal(self, root):
        def traversion(cur, rst):
            if cur == None:
                return
            traversion(cur.left, rst)
            rst.append(cur.val)
            traversion(cur.right, rst)
        rst = []
        traversion(root, rst)
        return rst

    def post_order_traversal(self, root):
        def traversion(cur, rst):
            if cur == None:
                return
            traversion(cur.left, rst)
            traversion(cur.right, rst)
            rst.append(cur.val)
        rst = []
        traversion(root, rst)
        return rst

    """迭代法"""
    """
    迭代法:
        递归的实现，就是每次把局部变量参数返回地址 压入调用栈中，递归返回时从栈顶弹出上次递归的参数。
        使用栈实现递归底层逻辑；
    先序遍历：
        要处理的和要访问的节点顺序是一致的，都是中间节点，
        处理中节点，现将根节点压入栈，处理中间节点，再将右节点压入栈，将左节点压入栈。
    中序遍历：
        要处理的（左）和要访问的（中）不一致，
        左中右，先访问左节点，直到最左边的节点，所以中序遍历，需要额外借助“指针来访问”节点，“栈来遍历”节点。
    后序遍历：
        前序遍历为中左右，后序遍历是左右中，只需调整先序遍历为：中右左，然后在反转result数组，就得到结果了；
    """
    def pre_order_traversal_iteration(self, root):
        if root == None:
            return root
        st = []
        st.append(root)
        rst = []
        while st:
            cur = st.pop()
            rst.append(cur.val)
            if cur.right:
                st.append(cur.right)
            if cur.left:
                st.append(cur.left)
        return rst

    def in_order_traversal_iteration(self, root):
        """
        中序遍历： 采用指针来处理数据，栈来遍历数据
        规则：
            while 栈不空 或者 cur 不为None：
                if cur != None:
                    说明还存在可以遍历的节点，不断将当前点入栈，然后继续遍历左子树
                else:
                    说明左子树为空，此时开始弹出cur上一个节点，处理cur，然后入栈右子树节点。重新循环
        :param root:
        :return:
        """
        rst = []
        st = [] # 栈用来遍历节点
        cur = root # 指针用来访问节点，访问底层节点
        while cur != None or st:
            if cur != None: # 不断压入堆栈，直到压倒最左边节点，让后进入else
                st.append(cur)
                cur = st.left
            else:
                cur = st.pop() # 没有左节点了，弹出当前节点处理，然后继续判断右节点的子树。
                rst.append(cur.val)
                cur = cur.right
        return rst

    def post_order_traversal_iteration_reverse(self, root):
        """
        后序遍历 采用转换顺序后的前序遍历，然后翻转result得到结果
        :param root:
        :return:
        """
        st = []
        rst = []
        st.append(root)
        while st:
            cur = st.pop()
            rst.append(cur.val)
            if cur.left: # 修改前序遍历的顺序，处理中节点，压入左节点，压入右节点。这样下次出栈就是右节点了（先进后出）。
                st.append(cur.left)
            if cur.right:
                st.append(cur.right)
        rst.reverse()
        return rst

    def post_order_traversal_iteration_java(self, root):
        """
        后序遍历：使用pre_visit标记已经访问过的节点，这种方法不太容易理解。
        规则：
            while 循环直接到达最左下节点
            while st:
                弹出cur
                如果cur有右节点，且 右节点没访问过；
                    压入右节点
                    cur = cur.right
                    while cur: # 遍历右子树
                        st 压入cur
                        cur = cur.left
                    继续遍历到右节点的最左边
                如果cur右节点为空，或者 右节点已经访问过
                    处理当前节点，
                    标记当前节点已经访问
        :param root:
        :return:
        """
        st = []
        rst = []
        cur = root
        pre_visit = None
        while cur != None:
            st.append(cur)
            cur = cur.left
        while st:
            cur = st.pop()
            if cur.right != None and cur.right != pre_visit:
                st.append(cur)
                cur = cur.right
                while cur != None:
                    st.append(cur)
                    cur = cur.left
            else:
                rst.append(cur)
                pre_visit = cur



    # 下面是二叉树统一迭代法：
    def in_order_traversal_iteration_unify(self, root):
        """
        迭代法中先序遍历借助栈实现遍历；中序遍历借助栈实现遍历，指针实现访问；后序遍历是先序遍历的一个变种；
        下面是二叉树统一迭代法：
            使用栈无法处理同时遍历节点和访问节点不一致的情况，如中序遍历；
            统一法使用栈访问节点，要处理的节点也放入栈中但是做个标记；标记方法：要处理的节点放入栈之后，紧接着放入一个空指针作为标记；（标记法）
        """
        st = []
        rst = []
        if root:
            st.append(root)
        while st:
            cur = st.pop()  # 弹出当前节点
            if cur != None: # 如果不为空，将所有节点压入栈进行遍历，依次压入 [右 中 None 左]
                if cur.right:
                    st.append(cur.right)
                st.append(cur)
                st.append(None)
                if cur.left:
                    st.append(cur.left)
            else:   # 为空，则说明下一个就是要处理的节点，弹出然后处理。
                cur = st.pop()
                rst.append(cur.val)
        return rst

    # 下面是二叉树统一迭代法：
    def pre_order_traversal_iteration_unify(self, root):
        st = []
        rst = []
        if root:
            st.append(root)
        while st:
            cur = st.pop()
            if cur != None:
                if cur.right:
                    st.append(cur.right)
                if cur.left:
                    st.append(cur.left)
                st.append(cur)
                st.append(None)
            else:
                cur = st.pop()
                rst.append(cur.val)
        return rst

    # 下面是二叉树统一迭代法：
    def post_order_traversal_iteration_unify(self, root):
        rst = []
        st = []
        st.append(root)
        while st:
            cur = st.pop()
            if cur:
                st.append(cur)
                st.append(None)
                if (cur.right):
                    st.append(cur.right)
                if cur.left:
                    st.append(cur.left)
            else:
                cur = st.pop()
                rst.append(cur.val)
        return rst


class LevelOrderTraversal:
    """
       层序遍历：
       102.二叉树的层序遍历
       107.二叉树的层次遍历II
       199.二叉树的右视图
       637.二叉树的层平均值
       429.N叉树的前序遍历
       515.在每个树行中找最大值
       116.填充每个节点的下一个右侧节点指针
       117.填充每个节点的下一个右侧节点指针II
       104.二叉树的最大深度
       111.二叉树的最小深度
           需要借助队列来实现，（栈适合模拟深度优先遍历的递归逻辑）
       """

    def level_order_traversal(self, root):
        results = []
        if not root:
            return results
        from collections import deque
        que = deque([root])
        while que:
            size = len(que)
            result = []
            for _ in range(size):
                cur = que.popleft()
                result.append(cur.val)
                # results.append(cur.val)
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            results.append(result)
        return results

    from collections import deque
    def levelOrderBottom(self, root):
        results = list()# []
        if not root:
            return results
        que = deque([root])
        while que:
            size = len(que)
            level = []
            for _ in range(size):
                cur = que.popleft()
                que.append(cur.val)
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            results.append(level)
        return results[::-1] ## 逆序输出

    """
    二叉树翻转：
        其实就是把每个节点的左右孩子交换一下就行；
        才有前序和后序都可以。
    """
    # 递归 ： 前中后序递归翻转
    def invertTree_recursion(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        # 先转换左右子节点
        root.left, root.right = root.right, root.left
        # 然后递归转换左右子节点
        self.invertTree_recursion(root.left)
        self.invertTree_recursion(root.right)
        return root
        #"""
        # invertTree(root->left); // 左
        # swap(root->left, root->right); // 中
        # invertTree(root->left); // 注意这里依然要遍历左孩子，因为中间节点已经翻转了
        # return root;
        #"""
    def invertTree_iterater(self, root):
        if root == None:
            return root
        st = list()
        st.append(root)
        while st != None:
            cur = st.pop()
            cur.left, cur.right = cur.right, cur.left
            if cur.right:
                st.append(cur.right)
            if cur.left:
                st.append(cur.left)
        return root

    def invertTree_level(self, root):
        que = deque()
        if root:
            que = deque([root])
        while que:
            size = len(que)
            for _ in range(size):
                cur = que.popleft()
                cur.left, cur.right = cur.right, cur.left
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
        return root



    """
    104 二叉树的最大深度
    给定一个二叉树，找出其最大深度。
    二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
    说明: 叶子节点是指没有子节点的节点。
    示例： 给定二叉树 [3,9,20,null,null,15,7]，
    思路和解法：
        递归法；迭代法；层序遍历法
        递归法：
            可以使用前序或者后续
            前序就是深度，后序就是高度；
            后序遍历：
                1. 确定递归函数参数和返回值：传入数的根节点，返回值就是数的深度；
                2. 终止条件，如果空节点的话，返回0，高度为0
                3. 单层逻辑：
                    先求左子树深度；
                    再求右子树深度；
                    去两者深度最大值 + 1（是因为算上当前中间节点） 这就是目前节点为根节点的数的深度
    
    """
    def maxDepth_post(self, root) -> int: # 递归法
        def get_depth(node):
            if node == None:
                return 0
            left = get_depth(node.left)
            right = get_depth(node.right)
            if node.left and not node.right:
                left += 1
            if not node.left and node.right:
                right += 1
            return max(left, right) + 1
        return get_depth(root)
    def maxDepth_post_pro(self, root) -> int:
        if root == None: return 0
        return max(self.maxDepth_post_pro(root.left), self.maxDepth_post_pro(root.right)+1)

    # 前序遍历 中左右， 这是求深度的逻辑
    def maxDepth_pre(self, root) -> int: # 递归法
        # result = 0
        def get_depth(node, depth):
            result = depth if (depth > result)  else result
            if not node.left and not node.right:
                return
            if node.left: #
                # depth +=1 # 深度+1
                # get_depth(node.left, depth)
                # depth -=1 # 回溯 深度-1
                get_depth(node.left, depth + 1)
            if node.right:
                # depth +=1
                # get_depth(node.right, depth)
                # depth -=1
                get_depth(node.right, depth + 1)
            return
        result = 0
        if root == None:
            return result
        get_depth(root, 1)
        return result

    def maxDepth_level(self, root) -> int: # 迭代法
        """
        使用迭代法的话，层序遍历最合适
        """
        depth = 0
        if root == None:
            return 0
        que = deque([root])
        while que:
            size = len(que)
            for _ in range(size):
                node = que.popleft()
                depth += 1
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
        return depth

    def maxDepth_post_n_tree(self, root) -> int:  # 递归法
        def get_depth(node):
            if node == None:
                return 0
            depth = 0
            for i in range(len(node.children)):
                depth = max(depth, get_depth(node.children[i]))
            # left = get_depth(node.left)
            # right = get_depth(node.right)
            # return 1+max(left, right)
            return depth
        return get_depth(root)


    """
    111. 二叉树的最小深度
    给定一个二叉树，找出其最小深度。
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    说明：叶子节点是指没有子节点的节点。
    思路和解法：
        1 迭代法 后续遍历 左右中 计算高度
            求二叉树的最小深度和求二叉树的最大深度的差别主要在于处理左右孩子不为空的逻辑。
    """
    def minDepth(self, root) -> int:
        def get_depth(root):
            # 如果为空，返回深度 0
            if root == None:
                return 0
            # 递归得到左右节点的深度
            left_depth= get_depth(root.left)
            right_depth = get_depth(root.right)
            # 判断本层的，那个节点存在就子节点深度加1
            if root.left == None and root.right != None:
                return right_depth+1
            if root.left != None and root.right == None:
                return left_depth+1
            return min(left_depth,right_depth) + 1
        return get_depth(root)

    def minDepth_iterater(self, root) -> int:
        """
        层序遍历迭代法：
        只有左右子孩都为空才为最低点；
        :param root:
        :return:
        """
        if root == None:
            return 0
        depth = 0
        que = deque([root])
        while que:
            size = len(que)
            depth+=1
            for _ in range(size):
                cur = que.popleft()
                if cur.left: que.append(cur.left)
                if cur.right: que.append(cur.right)
                if (not cur.left and not cur.right):
                    return depth
        return depth

    """
    222. 完全二叉树的节点个数
    给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
    完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
    思路和算法：
        普通二叉树：
            迭代法：层序遍历
                时间复杂度O(n) 空间复杂度O(n)
            递归法：后序遍历
                1 递归函数的参数（树的根节点）和返回值（根节点二叉树的节点数目）
                2 终止条件
                3 单层逻辑
                时间复杂度（n） 空间复杂度（logn 递归调用栈占用的空间）
        完全二叉树：
            利用完全二叉树的属性。满二叉树节点数2*depth-1;
            如果不是满二叉树，就递归其左右子孩，直到遇到满二叉树为止
                时间复杂度O(logn*logn) 空间复杂度O(logn)
    """
    def countNodes(self, root) -> int:
        def get_num(node) -> int:
            if node == None:
                return 0
            left = get_num(node.left)
            right = get_num(node.right)
            return left + right + 1
        return get_num(root)

    def countNodes_iterater(self, root) -> int:
        if root == None:
            return 0
        left_depth = 0
        right_depth = 0
        left = root.left
        right = root.right
        while left: # 求左子树深度
            left = left.left
            left_depth +=1
        while right:# 求右子树深度
            right = right.right
            right_depth +=1
        if left_depth == right_depth:
            # return pow(2,left_depth) -1
            return (2 << left_depth) - 1 #注意(2<<1) 相当于2^2，所以leftHeight初始为0
        return self.countNodes(root.left) + self.countNodes(root.right) + 1

    """
    110. 平衡二叉树
    给定一个二叉树，判断它是否是高度平衡的二叉树。
    本题中，一棵高度平衡二叉树定义为：
        一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
    思路和算法：
        二叉树节点的深度：指从根节点到该  节点的最长简单路径边数；深度需要从上到下所以需要前序遍历；
        二叉树节点的高度：指从该节点到叶子节点的最长简单路径边数；高度需要从下到上所以需要后续遍历
        注：leetcode中根节点深度为1；二叉树最大深度其实就是求根节点的高度所以可以用后序遍历；
        递归法：
            1 递归函数的参数（当前节点）和返回值（当前节点为根节点的数高度）
                如果当前节点为根节点的二叉树不符合平衡二叉树，就返回-1
            2 终止条件
            3 如果左右子树高度差小于等于1，返回高度；否则返回-1
    迭代法：
        层序遍历可以求深度，但是不能直接求高度；
        可以先通过栈模拟后序遍历找每一个节点的高度（其实是通过求传入节点为根节点的最大深度来求的高度）
        然后用栈来模拟前序遍历，遍历每个节点就判断左右子孩的高速是否符合
        这样写代码比较复杂，

    """
    def isBalanced(self, root) -> bool:
        def get_depth(node):
            if not node:
                return 0
            left = get_depth(node.left)
            if left == -1:
                return -1
            if (right := get_depth(node.right)) == -1:
                """
                python3 支持PEP572的海象运算符；
                把表达式的一部分赋值给变量，避免了两次求len(),多了一次赋值操作
                if (n := len(str) > 10):
                    return n
                n = len(a)
                    if n > 10:
                        print(f"List is to long({n} elements, expected <= 10)")
                """
                return -1
            if abs(left - right) > 1:
                return -1
            else:
                return max(left, right) + 1

        if get_depth(root) == -1:
            return False
        else:
            return  True
        # return False if (get_depth(root) == -1) else True

    def isBalanced_iterater(self, root) -> bool:
        def get_depth(node):
            st = []
            if node:
                st.append(node)
            depth = 0
            result = 0
            while st:
                cur = st.pop()
                if cur: # 后序遍历左右中  压栈顺序 中右左
                    st.append(cur)
                    st.append(None)
                    depth += 1
                    if cur.right: st.append(cur.right)
                    if cur.left : st.append(cur.left)
                else:
                    cur = st.pop()
                    depth -= 1
                result = max(result, depth)
        st = []
        if root:
            st.append(root)
        while st: # 前序遍历求深度中左右 入栈顺序右左中
            cur = st.pop()
            if abs(get_depth(cur.left)- get_depth(cur.right)) > 1:
                return False
            if cur.right:
                st.append(cur.right)
            if cur.left:
                st.append(cur.left)
        return True

    """
    101. 对称二叉树
    给定一个二叉树，检查它是否是镜像对称的。
    思路和解法：
        是否对称对比的可不是左右节点；而是根节点的左右子树是否互相翻转；
        所以递归遍历的过程需要同时遍历两个子树。注意需要外侧和外侧相比较，所以采用后序遍历【左右中】，一颗为【左右中】一颗为【右左中】
        注：后序遍历可以理解为一种回溯；
        使用队列来存储接下来需要对比的节点，类似于层序遍历；
    """

    def isSymmetric_recursion(self, root: TreeNode) -> bool:
        if not root: return True;

        def compare(left, right):
            # 终止条件
            if (not left and right) or (left and not right):
                return False
            elif not left and not right:
                return True
            elif left.val != right.val:
                return False
            isleft = compare(left.left, right.right)
            isright = compare(left.right, right.left)
            if isright and isleft:
                return True
            else:
                return False

        return compare(root.left, root.right)

    def isSymmetric_iterator(self, root: TreeNode) -> bool:
        """
        使用队列比较两个树能否互相翻转
        :param root:
        :return:
        """
        if not root: return True
        # st = []
        # st.append(root.left)
        # st.append(root.right)
        # while st:
        #     right = st.pop()
        #     left = st.pop()
        #     if not left and not right:
        #         continue
        #     if not left or not right or (left.val != right.val):
        #         return False
        #     st.append(left.left)
        #     st.append(right.right)
        #     st.append(left.right)
        #     st.append(right.left)
        # return True

        que = deque()
        que.append(root.left)
        que.append(root.right)
        while que:
            left = que.popleft()
            right = que.popleft()
            if not left and not right:
                continue
            if not left or not right or (left.val != right.val):
                return False
            que.append(left.left)
            que.append(right.right)
            que.append(left.right)
            que.append(right.left)
        return True



    """
    257. 二叉树的所有路径
    给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。
    叶子节点 是指没有子节点的节点。
    思路和解法：
        根节点到叶子节点 - 前序遍历（中左右） - 回溯
        递归法：
            1 递归函数参数（当前节点，路径，result）和返回值（此处只需要放入结果集result不用返回值）
            2 终止条件：到达叶子节点停止，如何判断叶子节点：【左右子节点均不存在】回溯
            3 单层逻辑： （path和result都随着递归传递）
                1 前序遍历，先处理中间节点，将当前cur.val 加入path
                2 判断是否结束（终止条件）结束的话将当前path加入result
                3 递归和回溯
                  如果左子节点不为空，遍历左节点
                  如果右子节点不为空，遍历右节点
        迭代法：
            使用前序遍历的迭代法模拟遍历路径过程，借助栈
            
    """
    def binaryTreePaths_backtracking(self, root):
        def traversal(cur, path, result):
            path.append(cur.val)
            if cur.left == None and cur.right == None:
                spath = ''
                for i in path:
                    spath += str(i)
                result.append(spath)
                return
            if cur.left:
                traversal(cur.left, path, result)  # 递归
                path.pop()                         # 回溯 回溯一定要跟着递归，在一个代码块中
            if cur.right: # 注意下面的这个path还是当前递归的path，而不是left之后的，所以不需要回溯
                traversal(cur.right, path, result)
                path.pop()
        result = []
        path = []
        if not root:return result
        traversal(root, path, result)
        return result
    """
    区分binaryTreePaths_backtracking 和 binaryTreePaths，两者的traversal前者传入path为list，后者传入的为str
    后者的string path 每次都是赋值值，不用使用引用，否则就无法得到回溯效果了；
    而且后者的每次traversal中 调用完path，其实并没有加上递归中的参数
    """
    def binaryTreePaths(self, root):
        def traversal(cur, path, result):
            path += str(cur.val)
            if cur.left == None and cur.right == None:
                result.append(path)
            if cur.left:
                traversal(cur.left, path, result)
            if cur.right: # 注意下面的这个path还是当前递归的path，而不是left之后的，所以不需要回溯
                traversal(cur.right, path, result)
        result = []
        path = ''
        if not root:return result
        traversal(root, path, result)
        return result

class TreeCreate:
    """
    构造二叉树
    """
    """
    106. 从中序与后序遍历序列构造二叉树
    根据一棵树的中序遍历与后序遍历构造二叉树。
    注意:
        你可以假设树中没有重复的元素。
        例如，给出
        中序遍历 inorder = [9,3,15,20,7]   左中右
        后序遍历 postorder = [9,15,7,20,3] 左右中
    思路和算法：
        以后续数组的最后一个元素为切割点，先切中序数组，根据中序数组，反过来在切后序数组，一次次切割，每次后序数组最后一个元素就是节点元素。
        代码逻辑：
            1 如果数组为0 则为空节点
            2 如果不为空，去后序数组的最后一个元素作为节点元素
            3 找到节点元素在中序数组中的位置，作为切割点，切割中序数组，得到中序左数组，中序右数组；
            4 切割后序数组，得到后续左数组，后序右数组；
            5 递归处理左右区间
    """
    def buildTree_in_post(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not postorder:
            return None
        root_value = postorder[-1]
        root = TreeNode(root_value)

        # if len(postorder) == 1:return root

        # 中序数组中找到切割点
        index = inorder.index(root_value)
        # index = 0
        # for index in range(len(inorder)):
        #     if inorder[index] == root_value:
        #         break
        # 切割中序数组
        left_inorder = inorder[:index]
        right_inorder = inorder[index+1:]

        # 切割后序数组 重点 中序数组一定要跟后续数组大小相同
        left_postorder = postorder[0:len(left_inorder)]
        right_postorder = postorder[len(left_inorder): len(postorder) - 1]
        # 递归左右子树
        self.buildTree_in_post(left_inorder, left_postorder)
        self.buildTree_in_post(right_inorder, right_postorder)
        return root


    def buildTree_pre_in(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        root_val = preorder[0]
        root = TreeNode(root_val)

        index = inorder.index(root_val)

        left_inorder = inorder[:index]
        right_inorder = inorder[index+1:]

        left_preorder = preorder[1:len(left_inorder) + 1]
        right_preorder = preorder[len(left_inorder)+1:]
        self.buildTree_pre_in(left_preorder,left_inorder)
        self.buildTree_pre_in(right_preorder,right_inorder)
        return root # 递归传入的是对象，

    """
    654. 最大二叉树
        给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：
        
        二叉树的根是数组 nums 中的最大元素。
        左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
        右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
        思路和算法：
            递归法：
                1 参数（num）和返回值（本次递归要添加的节点）
                2 终止条件：数组大小为1时，为叶子节点。就定义一个新节点，返回。
                3 单层递归逻辑：
                    找到数组中最大的值和索引，新建节点root；
                    如果索引>0 构造左子树
                    如果索引《len(nums)-1 构造右子树
    """
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        """
        这样写代码比较冗余，效率不高。每次需要定义新的数组存储切割得到的数组，但是逻辑比较清晰。-
        :param self:
        :param nums:
        :return:
        """
        node = TreeNode(0)
        if len(nums) == 1:
            nums.val = nums[0]
            return node
        max_val = 0
        max_index = 0
        for i in range(len(nums)):
            if nums[i] > max_val:
                max_val = nums[i]
                max_index = i
        root = TreeNode(max_val)
        if max_index > 0:
            root.left = constructMaximumBinaryTree(nums[:max_index])
        if max_index < len(nums) - 1:
            root.right = constructMaximumBinaryTree(nums[max_index+1:])
        return root

    def constructMaximumBinaryTree_pro(self, nums: List[int]) -> TreeNode:
        def constructMaximumBinaryTree(nums: List[int], start, end) -> TreeNode:
            # 列表长度为0，返回空
            if start == end:
                return None

            # 找最大值和下标
            max_index = start
            for i in range(start, end):
                if nums[i] > nums[max_index]:
                    max_index = i
            # 构建当前节点
            root = TreeNode(nums[max_index])
            # 递归构建左右子树
            constructMaximumBinaryTree(nums, start, max_index)
            constructMaximumBinaryTree(nums,max_index+1, end)
            return root
        return constructMaximumBinaryTree(nums, 0, len(nums))

    """
    617. 合并二叉树
    给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
    你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，
    否则不为 NULL 的节点将直接作为新二叉树的节点。
    思路和算法：
        遍历树，同时传入两棵树同时操作。
        递归（三种都可以，这里使用前序遍历）：
            1. 参数两棵树的root，返回值就是合并之后的节点
            2. 终止条件：如果两个数都为空，
            3. 单层逻辑：
        迭代法：
            使用队列模拟层序遍历
    扩展：和对称二叉树类型
    """
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        # 如果两者中某个为空，直接返回另一个就行。
        if not root1: return root2
        if not root2: return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1 # 本题我们重复使用了题目给出的节点而不是创建新节点. 节省时间空间
    def mergeTrees_iterater(self, root1, root2):
        if not root1: return root2
        if not root2: return root1
        que = deque()
        que.append(root1)
        que.append(root2)
        while que:
            # 弹出两棵树的root节点
            node1 = que.popleft()
            node2 = que.popleft()
            # 使用root1，作为原来的数。
            node1.val += node2.val
            # 如果左右节点都不为空，就入栈，然后继续迭代
            if node1.left and node2.left:
                que.append(node1.left)
                que.append(node2.left)
            if node1.right and node2.right:
                que.append(node1.right)
                que.append(node2.right)
            # 如果node1为空，就重新定义node1的指针
            if not node1.left and node2.left:
                node1.left = node2.left
            if not node1.right and node2.right:
                node1.right = node2.right
        return root1

class TreeSearch:
    """
    二叉搜索树是一个有序树：
        左子树不为空 左子树上节点值均小于根
        右子树不为空 右子树上节点值均大于根
        左右子树分别为二叉搜索树
    这就决定了二叉搜索树的，递归遍历和迭代遍历都不一样；
    二叉搜索树中搜索
    是不是二叉搜索树
    二叉搜索树的最小绝对差
    二叉搜索树的众数
    """
    """
    700. 二叉搜索树中的搜索
    给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
    思路和算法：
        递归法：
            1 参数为二叉树节点和要查找的值， 返回值为目标节点
            2 if not root || root.val == val: return root
            3 单层递归逻辑：
                 如果root.val > val 搜索左子树；反之亦然
            4 如果都没搜索到就返回Null
        注：什么时候需要return什么时候不需要，如果搜索某个值，就要立即return，这样才能不遍历整个树
        迭代法：
            提到二叉树遍历，可以使用栈来模拟深度遍历，使用队列模拟广度优先遍历；但是二叉搜索树本身就是有序的不需要栈和队列辅助；
            递归过程中有回溯过程，如果左分支走到头就调用，回溯。而二叉搜索树不需要回溯。
    """
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root or root.val == val:
            return root
        if root.val > val:
            return self.searchBST(root.right, val)
        if root.val < val:
            return self.searchBST(root.left, val)
        return None

    def searchBST_iterator(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if root.val == val:
                return root
            elif root.val > val:
                root == root.left
            elif root.val < val:
                root == root.right
        return None

    """
    98. 验证二叉搜索树
    给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
    有效 二叉搜索树定义如下：
    节点的左子树只包含 小于 当前节点的数。
    节点的右子树只包含 大于 当前节点的数。
    所有左子树和右子树自身必须也是二叉搜索树。
    思路和解法：
        中序遍历（左中右）得到的数值就是有序的。
        递归法：
            中序法得到数组，判断数组是否有序；
            注意：不能每次递归比较左中右三者的顺序，因为我们要求是左边的节点均小于根节点；
        
    """

    def isValidBST(self, root: TreeNode) -> bool:
        """
        中序遍历将节点值压入数组中
        :param root:
        :return:
        """
        list_s = []
        def get_list:
            if not root:return
            self.isValidBST(root.left)
            self.list_s.append(root.val)
            self.isValidBST(root.right)
        get_list(root)
        for i in range(len(list_s) - 1):
            if list_s[i] >= list_s[i+1]:
                return False
        return True

    def isValidBST_biaozhun(self, root: TreeNode) -> bool:
        """
        BST的中序遍历节点数值从小到大
        :param root:
        :return:
        """
        cur_max = -float("INF") # 设置一个极小值
        pre = TreeNode(None) # 如果不设置极小值，可以定义一个节点存储前一个节点的值，只要比前一个节点大就行
        def _isValidBST(root):
            nonlocal cur_max
            if not root:
                return True
            # 中序遍历 验证遍历的元素是否从小到大
            is_left = _isValidBST(root.left)
            if cur_max < root.val:
                cur_max = root.val
            else:
                return False
            # if not pre and pre.val < root.val:  # 如果不设置极小值，
            #     pre.val = root.val
            # else:
            #     return False

            is_right = _isValidBST(root.right)

            return is_left and is_right
        return _isValidBST(root)

    def isValidBST_iterator(self, root: TreeNode) -> bool:
        """
        使用栈模拟中序遍历
        中序遍历的迭代写法：1用cur和st判断;来实现遍历到最左边节点。
        :param root:
        :return:
        """
        st = list()
        cur = root
        pre = None
        while st != None or cur != None:
            if cur != None:
                st.append(cur)
                cur = cur.left
            else:
                cur = st.pop()
                if pre != None and pre.val >= cur.val:
                    return False
                pre = cur
                cur = cur.right
        return True

    """
    530. 二叉搜索树的最小绝对差
    给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
    差值是一个正数，其数值等于两值之差的绝对值。
    思路和算法：
        
    """

    """"
    701. 二叉搜索树中的插入操作
    给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。
    返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。
    注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。

    注： 
        搜索树最小绝对差和众数 使用pre和cur两个指针，迭代法同理
    """
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if root == None:
            node = TreeNode(val)
            return node
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        return root
    def insertIntoBST_iterator(self, root: TreeNode, val: int) -> TreeNode:
        if root == None:
            node = TreeNode(val)
            return node
        cur = root
        pre = root # 中要，需要记录上一个节点，否则无法赋值新节点
        while cur:
            pre = cur
            if cur.val > val:
                cur = cur.left
            else:
                cur = cur.right
        node = TreeNode(val)
        if val < pre.val:
            pre.left = node
        else:
            pre.right = node
        return root


class commenForther:
    """
    236. 二叉树的最近公共祖先
    给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
    百度百科中最近公共祖先的定义为：
        “对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

    思路和算法：
        1 找最小祖先，自底向上查找就可以，如何自下向上？ 后序遍历就是天然的自底向上；
        2 回溯中必须要找整个二叉树，因此递归的函数需要返回值，进行判断；
        3 单层逻辑：
            先判断终止条件：
                如果找到了，就返回节点（说明找到了），如果为空也返回（说明到底了）
            遍历左右；
            判断当前节点：
                如果左右都不为空，那么说明左右子树都找到了； 直接返回root
                如果某一边为空，就返回不为空的节点；
                否则返回None（表示当前子树没找到，需要继续回溯）
    注：
        后序遍历就是天然的自底向上；
    """

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 如果找到p q 或者空节点 返回
        if root == p or root == q or root == None:
            return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)

        if left and right:
            return root
        if left == None and right != None:
            return right
        elif left != None and right:
            return left
        else:
            return None

if __name__ == "__main__":
    rst = [1,2,3,4,5]
    eg = [3,2,1,6,0,5]
    exam_tc = TreeCreate()
    result = exam_tc.constructMaximumBinaryTree_pro(rst)
    print(result)