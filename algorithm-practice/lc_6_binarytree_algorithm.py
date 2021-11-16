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



class TreeNode:
    def __init__(self, val):
        left = None
        right = None
        self.val= val

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
    def maxDepth_post(self, root) -> int:
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
    def maxDepth_pre(self, root) -> int:
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

    def maxDepth_level(self, root) -> int:
        """
        使用迭代法的话，层序遍历最合适
        """
        if 

    """
    111. 二叉树的最小深度
    给定一个二叉树，找出其最小深度。
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    说明：叶子节点是指没有子节点的节点。
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

if __name__ == "__main__":
    rst = [1,2,3,4,5]
    print(rst[:1])

