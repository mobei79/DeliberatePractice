# -*- coding: utf-8 -*-
"""
@Time     :2022/2/9 14:47
@Author   :jingjin.guo@fengjr.com
@Last Modified by:
@Last Modified time:
@file :
@desc :
"""

"""
1.1 广义线性模型
    logistic：‘目标值y’是‘输入变量x’的线性组合【即只有加权累加】
    可以用于回归或者分类的方法
        向量权重 coef_ ;偏置bias intercept_
    1.1.1 普通的最小二乘法
        LinearRegression 拟合的是一个带系数w的线性模型，使用数据集‘实际观测数据’和‘预测数据’之间的残差平方和最小。
        LinearRegression调用fit方法来拟合数组X,y，并将线性模型的系数w存储在其‘成员变量’coef_中
        普通最小二乘法的系数估计问题，依赖于模型各项之间相互独立性。当各项相关时，且设计矩阵X的各列近似线性相关，那么设计矩阵会趋向于奇异矩阵，这会导致最小二估计对于随机误差非常敏感，产生大的方差。
        最小二乘法的复杂度：使用X的奇异值分解来计算最小二乘解。如果X是一个size为(n,p)的矩阵，假设n>p，则该方法的复杂度为O(n*P^2)
    1.1.2 Ridge 岭回归 L2正则化
        Ridge 回归通过对系数的大小施加惩罚解决普通最小二乘法的一些问题，岭系数最小化的是带惩罚的残差平方和，α越大收缩量越大，系数对共线性的鲁棒性越强。
        岭回归复杂度和普通最小二乘法相同的。
        如何设置正则化参数？
            广义交叉验证：RidgeCV 通过内置的 Alpha 参数的交叉验证来实现岭回归。 
            该对象与 GridSearchCV 的使用方法相同，只是它默认为 Generalized Cross-Validation(广义交叉验证 GCV)，这是一种有效的留一验证方法（LOO-CV）:
    1.1.3 Lasso L1正则化 
        Lasso是估计稀疏系数的线性模型。倾向于使用较少参数值的情况，有效地较少给定解决方案所依赖的变量的数量。Lasso及其变体是压缩感知领域的基础。
        在数学上，Lasso是由一个带有L1先验的正则项的线性模型组成。产生稀疏模型，可以用于特征选择。
        Lasso类的实现使用 coordinate descent（坐标下降算法）来拟合系数。【也可以采用最小角回归】
        如何设置正则化参数？
            可以使用LassoCV或者LassoLarsCV【基于最小角回归算法，对于高维数据集此交叉验证最快】
        alpha和SVM正则化参数比较？
            关系式为：alpha=1/C 或者alpha=1/(n_samples*C)
    1.1.4. 多任务 Lasso
        MultiTaskLasso 是一个估计多元回归稀疏系数的线性模型： y 是一个 (n_samples, n_tasks) 的二维数组，其约束条件和其他回归问题（也称为任务）是一样的，都是所选的特征值。
    1.1.5. 弹性网络
        是一种使用L1 L2范数作为先验正则项训练的线性回归模型。
    1.1.6. 多任务弹性网络
        MultiTaskElasticNet 是一个对多回归问题估算稀疏参数的弹性网络: Y 是一个二维数组，形状是 (n_samples,n_tasks)。 其限制条件是和其他回归问题一样，是选择的特征，也称为 tasks 。
    1.1.7. 最小角回归 LARS
        优点：
            当 p >> n，该算法数值运算上非常有效。(例如当维度的数目远超点的个数)
            它在计算上和前向选择一样快，和普通最小二乘法有相同的运算复杂度。
            它产生了一个完整的分段线性的解决路径，在交叉验证或者其他相似的微调模型的方法上非常有用。
            如果两个变量对响应几乎有相等的联系，则它们的系数应该有相似的增长率。因此这个算法和我们直觉 上的判断一样，而且还更加稳定。
            它很容易修改并为其他估算器生成解，比如Lasso。
        缺点：
            因为 LARS 是建立在循环拟合剩余变量上的，所以它对噪声非常敏感。
            
    1.1.8. LARS Lasso
        LassoLars 是一个使用 LARS 算法的 lasso 模型，不同于基于坐标下降法的实现，它可以得到一个精确解，也就是一个关于自身参数标准化后的一个分段线性解。
    1.1.9. 正交匹配追踪法（OMP）
        OrthogonalMatchingPursuit (正交匹配追踪法)和 orthogonal_mp
        使用了 OMP 算法近似拟合了一个带限制的线性模型，该限制影响于模型的非 0 系数(例：L0 范数)。
    1.1.10. 贝叶斯回归
        贝叶斯回归可以用于在预估阶段的参数正则化: 正则化参数的选择不是通过人为的选择，而是通过手动调节数据值来实现。
        贝叶斯岭回归:BayesianRidge 利用概率模型估算了上述的回归问题，其先验参数 w 是由以下球面高斯公式得出的
            ARDRegression （主动相关决策理论）和 Bayesian Ridge Regression_ 非常相似，
    1.1.11. logistic 回归
        logistic回归，虽然名字里有 “回归” 二字，但实际上是解决分类问题的一类线性模型。
        在某些文献中，logistic 回归又被称作 logit 回归，maximum-entropy classification（MaxEnt，最大熵分类），或 log-linear classifier（对数线性分类器）。该模型利用函数 logistic function 将单次试验（single trial）的可能结果输出为概率。
        LogisticRegression 类中实现了二分类（binary）、一对多分类（one-vs-rest）及多项式 logistic 回归，并带有可选的 L1 和 L2 正则化
        LogisticRegression中实现了这些优化算法（也叫求解器）【liblinear、newton-cg、lbfgs、sag、saga】
        选用求解器可遵循如下规则:
            L1正则	“liblinear” or “saga”
            多项式损失（multinomial loss）	“lbfgs”, “sag”, “saga” or “newton-cg”
            大数据集（<cite>n_samples</cite>）	“sag” or “saga”
        “saga” 一般都是最佳的选择，但出于一些历史遗留原因默认的是 “liblinear” 。
        对于大数据集，还可以用 SGDClassifier ，并使用对数损失（’log’ loss）
        L1惩罚会达到特征选择的作用；
        LogisticRegressionCV 对 logistic 回归 的实现内置了交叉验证（cross-validation），可以找出最优的参数 C 。”newton-cg”， “sag”， “saga” 和 “lbfgs” 在高维数据上更快，因为采用了热启动（warm-starting）。 在多分类设定下，若 multi_class设为 “ovr” ，会为每类求一个最佳的 C 值；若multi_class设为 “multinomial” ，会通过交叉熵损失（cross-entropy loss）求出一个最佳 C 值。
    1.1.12 随机梯度下降， SGD
        梯度下降是拟合线性模型的简单有效的方法，样本量大，特征数大时尤为有效。
        partial_fit可用于在线学习（online learning）或基于外存的学习（out of core learning）
        SGDClassifier和SGDRegressor分别用于拟合分类问题和回归问题的线性模型，支持不同的损失函数：loss="log"拟合逻辑回归模型，loss="hinge"拟合线性支持向量机svm； 
    1.1.13. Perceptron（感知器）
        Perceptrom （linear_model.Perceptron）是适用于大规模学习的一种简单算法。
        不需要设置学习率（learning rate）；不需要正则化处理；仅使用错误样本更新模型。
        最后一点表明使用合页损失（hinge loss）的感知机比 SGD 略快，所得模型更稀疏。
    1.1.14. Passive Aggressive Algorithms（被动攻击算法）
        被动攻击算法是大规模学习的一类算法。和感知机类似，它也不需要设置学习率，不过比感知机多出一个正则化参数 C 。
        对于分类问题， PassiveAggressiveClassifier 可设定 loss='hinge' （PA-I）或 loss='squared_hinge' （PA-II）。对于回归问题， PassiveAggressiveRegressor 可设置 loss='epsilon_insensitive' （PA-I）或 loss='squared_epsilon_insensitive'
    1.1.15.4. Huber 回归
        HuberRegressor 与 Ridge 不同，因为它对于被分为异常值的样本应用了一个线性损失。如果这个样品的绝对误差小于某一阈值，样品就被分为内围值。 它不同于 TheilSenRegressor 和 RANSACRegressor ，因为它没有忽略异常值的影响，并分配给它们较小的权重。
            
            
"""
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])



"""
1.6 最近邻
    sklearn.neighbors 提供了neighbors-based(基于邻居的)无监督及监督学习功能。
    无监督的最近邻是许多其他学习方法的基础，尤其是manifold learning（流行学习）和spectral clustering（谱聚类）。许多学习方法依赖最近邻作为核心，如“核密度估计”
    分为classification（针对具有离散标签的数据）；regression（针对具有连续标签的数据）。 
    Neighbors-based方法被称为“非泛化机器学习方法”，因为只是简单“记住”所有的训练数据（可能转化为一个快速索引结构，如bell-tree或者kd-tree）
    Neighbors-based为非参数化方法(non-parametic)，尽管简单，但是最近邻算法是用于很多分类和回归问题。如手写数字或者卫星图像场景。经常成功应用于决策边界非常不规则的分类情况下。
    sklearn.neighbors可处理Numpy数组或者scipy.sparse矩阵。
    
    基本原理：
        从训练样本中找到与新点距离最近的预订数量的几个点，然后从这些点中预测标签。
        k可以用户定义（K-近邻）；也可根据不同点的局部密度（基于半径的最近邻）。
        距离度量通常使用standard标准欧氏距离。
    
    1.6.1 无监督最近邻
        NearestNeighbors实现了无监督最近邻学习，为三种不同的最近邻算法提供了统一的接口：Ball tree/KD tree/brute-force。通过algorithm控制【可选auto、ball_tree、kd_tree、brute】
        简单任务可以使用sklearn.neighbor中的无监督算法
        复杂的可以使用KDTree和BallTree类，具有相同的接口
    1.6.2 最近邻分类
        最近邻分类属于“基于实例的学习 or 非泛化学习”。他不会构建一个泛化的内部模型，而是简单的存储训练数据的实例。有每个点最近邻的投票决定。
        KNeighborsClassifier 基于查询点的k个最近邻
        RadiusNeighborsClassifier 基于查询点的固定半径r内的邻居数决定
        KNeighborsClassifier是最常用的，最佳值k高度依赖于数据，较大的k会抑制噪声，但是会使得分类界面不明显；如果数据是不均匀采样的（某类非常多，某类非常少），那么RadiusNeighborsClassifier中基于r的分类更好，但是高纬度数据会造成“维度灾难”。
        weight控制权重，可选值【uniform distance】
    1.6.3 最近邻回归
        数据标签为连续变量，分配给查询点的标签是由它的邻近标签的均值计算而来。
        scikit-learn中有两种不用的最近邻回归：
            KNeighborsRegressor基于每个检查点的k个近邻实现；
            RediusNeighborsRegressor基于每个检查点的固定半径r内的邻点数量竖线。
        最基本的最近邻回归使用统一的权重；也可使用weights关键字加权：“uniform”等权重；‘distance’权重为距离反比；也可以自定义；
    
    1.6.4 最近邻算法
        1. 暴力计算，涉及到数据集中所有成对点之间距离的暴力计算，对于D维度中N个样本，查询所有成对点之间的距离，算法复杂度为O(D*N^2)。
            适用于小数据样本。暴力计算关键字algorithm='brute'
        为了解决暴力计算效率低下的问题，发明了很多基于树的数据结构，通过有效编码样本的聚合距离（aggregate distance）信息来减少所需的距离计算量。计算成本降低到O(DNlog(N))或者更低。
        2. KD树，使用kdtree数据结构，构建快（因为只需沿数据周执行分区）一旦构建完成，查询点的最近邻距离计算复杂度为O(log(N))；
            对于低纬度（D<20）非常快，随着D增大效率降低即所谓的维度灾难。关键字algorithm=‘kd_tree’；
        3. bell树，构建起来比KD树更耗时，但是数据结构对于高结构化的数据非常有效，高维度也有效。
            性能高度依赖训练数据的结构。
        4 最近邻算法的选择
            对给定数据集选择最优算法是复杂的过程，取决于多个因素。
            样本数量N（n_samples）、维度D（n_features），kdtree和belltree通过leaf_size参数控制小数据及是计算效率；
            数据结构：数据的本征维度（数据所在的流行的维度 d<=D）；数据的稀疏度（数据填充参数空间的程度，不用于稀疏矩阵）
                brute force不受数据结构影响；
                ball tree和KDtree受影响较大，一般，小维度的系数数据会使得查询更快；kd树不如ball tree效果好。
            查询点（query point）所需的近邻数K
                brute force查询时间几乎不受k影响。ball和kd会随着k增加变慢。
            查询点数：ball和kd构建需要时间，查询量大时，构造成本就分摊了忽略不计。

    algorithm = 'auto' 算法会尝试从训练数据中确定最佳方法。
    leaf_size
        其默认值 30.
        小样本暴力搜索比基于树的搜索方法更加有效，ball和kd树中使用参数leaf_tree指定。
        构造时间:更大的 leaf_size 会导致更快的树构建时间, 因为需要创建更少的节点.
        查询时间:一个大或小的 leaf_size 可能会导致次优查询成本. 当 leaf_size 接近 1 时, 遍历节点所涉及的开销大大减慢了查询时间. 当 leaf_size, 接近训练集的大小，查询变得本质上是暴力的. 这些之间的一个很好的妥协是 leaf_size = 30, 这是该参数的默认值.
"""

"""
1.2 线性判别分析（LDA）和二次判别分析（QDA）
    discriminant_analysis.LinearDiscriminantAnalysis 
    discriminant_analysis.QuadraticDiscriminantAnalysis) 是两个经典的分类器，分别代表线性决策平面和二次决策平面
    这些分类器，很容易计算得到解析解；具有分类的特性；无需调参；
1.3. 内核岭回归
    KernelRidge学习模型与支持向量回归SVR一样。
    但是使用不同的损失函数：
        内核岭回归（KRR）使用squared error loss(平方误差损失函数)
        support vector regression（支持向量回归）使用e-insensitive loss（不敏感损失）
    两者都是用L2正则化。
1.4 支持向量机 （svms）
    可以用于分类 回归 异常检测，监督学习
    优势：
        高维空间非常高效；数据维度比样本数量大的情况下有效；高效利用内存，在决策函数（称为支持向量）中使用训练集的子集；不同的核函数与特定的决策函数一一对应。
    劣势：
        如果特征数量比样本数量大得多,在选择核函数 核函数 时要避免过拟合，需要正则化项；SVM不支持提供概率估计；
    1.4.1 分类
        SVC、NuSVC、LinearSVC可实现多元分类
        SVC 和 NuSVC 是相似的方法, 但是接受稍许不同的参数设置并且有不同的数学方程；
        LinearSVC 是另一个实现线性核函数的支持向量分类. 记住 LinearSVC 不接受关键词 kernel, 因为它被假设为线性的. 它也缺少一些 SVC 和 NuSVC 的成员(members) 比如 support_ .
        输入：和其他分类器一样, SVC, NuSVC 和 LinearSVC 将两个数组作为输入: [n_samples, n_features] 大小的数组 X 作为训练样本, [n_samples] 大小的数组 y 作为类别标签(字符串或者整数):
        多元分类：
        得分和概率：SVC 构造器选项probability=‘True’开启可能性评估，在训练集上使用额外的交叉验证来拟合；建议设置为Flase
        关键参数：
            kernel
            clf.support_vectors_（获取支持向量）; support_（支持向量的索引）、n_support_(每个类别获得支持向量的数量)
    
    1.4.2 回归
        支持向量机只依赖于训练集的子集，因为构建模型的loss function不在乎边缘之外的训练集。
        sklearn中有三种形式：SVR、NuSVR、LinearSVR。
    1.4.4 复杂度        
        SVM需要的计算和存储空间 随着训练向量的数目增加而快速增加，SVM核心是一个二次规划问题。
    1.4.5 使用诀窍
        避免数据复制
            
            
“”“

“”“
1.5 随机梯度下降
   SGD 主要用于凸损失函数下线性分类器的判别式学习。如SVM和逻辑回归。
   SGD可应用于大规模稀疏机器学习问题。本模型可轻易处理超过10^5的训练样本和10^5的特征；
   优点：高效、易实现；缺点：需要一些差参数，如正则化参数和迭代次数、对特征缩放敏感。
   1.5.1 分类
   SGDClassifier类实现了简单的随机梯度下降，支持不同的loss function和分类惩罚；
   
   参数：
    使用shuffle=True打乱训练数据；
    loss参数可选区间：
        loss="hinge": (soft-margin) linear Support Vector Machine （（软-间隔）线性支持向量机），
        loss="modified_huber": smoothed hinge loss （平滑的 hinge 损失），
        loss="log": logistic regression （logistic 回归），and all regression losses below（以及所有的回归损失）。
    前两个 loss functions（损失函数）是懒惰的，如果一个例子违反了 margin constraint（边界约束），它们仅更新模型的参数, 这使得训练非常有效率,即使使用了 L2 penalty（惩罚）我们仍然可能得到稀疏的模型结果。
    使用 loss="log" 或者 loss="modified_huber" 来启用 predict_proba 方法, 其给出每个样本 x 的概率估计 P(y|x) 的一个向量：
    penalty参数设定惩罚类型：
        penalty="l2": L2 norm penalty on coef_.
        penalty="l1": L1 norm penalty on coef_.
        penalty="elasticnet": Convex combination of L2 and L1（L2 型和 L1 型的凸组合）; (1 - l1_ratio) * L2 + l1_ratio * L1.
    average=True参数设置是否均值化。AGSD 工作原理是在普通 SGD 的基础上，对每个样本的每次迭代后的系数取均值。当使用 ASGD 时，学习速率可以更大甚至是恒定，在一些数据集上能够加速训练过程。
    多分类参数：
        SGDClassifier通过“one versus all”(OVA)方法组合多个二分类器实现多分类。选择置信度最高的分类。
        多分类情况下。 clf.coef_ 是shape=[n_class, n_features]的二维数组。clf.intercept_是shape=[n_class]的一维数组。
        loss='log' 和“modified_huber” 更适合OVA分类
        原则上，允许创建一个概率模型。

    1.5.2 回归
        SGDRegressor适合大量训练样本（》10000）的回归问题，其他问题推荐使用Ridge/Lasso/ElasticNet
        loss参数：
            loss="squared_loss": Ordinary least squares（普通最小二乘法）,
            loss="huber": Huber loss for robust regression（Huber回归）,
            loss="epsilon_insensitive": linear Support Vector Regression（线性支持向量回归）.
    补充     
        随机梯度下降法对 feature scaling(特征缩放)很敏感，建议缩放数据(即标准化 StandardScalar)。
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
        最好使用GridSearchCV找到合理的regularization term正则化参数，通常 = 10.0**-np.arange(1,7)
        经验表明SGD在处理10^6训练样本后基本收敛。迭代次数n_iter = np.ceil(10**6 / n)，其中 n 是训练集的大小
“”“


“”“

1.7 高斯过程
    Gaussian Process GP，是概率论和数理统计中随机过程中的一种，是一系列服从正态分布的随机变量在一指定集内的组合。
    GP是一种常用的监督学习方法，旨在解决回归问题和概率分类问题。
        优点：
            预测内插了观察结果；预测结果是概率形式；可以指定不同的内核；
        缺点：
            他们不稀疏，通常使用整个样本的特征信息进行预测；高纬度（特征数超过几十个）失效；

1.8 交叉分解
    交叉分解模块主要包含两个算法族：偏最小二乘法（PLS）和典型相关分析（CCA）。
    这些算法具有发现两个多远数据集之间线性关系的用途：fit method的参数x和y都是二维数组。
    交叉分解能够找到两个矩阵（X，Y）的基础关系。他们是对两个空间的协方差结构进行建模的隐变量方法。尝试在X空间中找到多维方向，该方向能够解释Y空间中最大多维方差方向。
    PLS回归是用于：预测变量矩阵具有比观测值更多的变量，以及当X值存在多重共线性时。

1.9 朴素贝叶斯
    朴素贝叶斯方法是基于贝叶斯定理的一组有监督学习算法，即“简单”地假设每对特征之间相互独立。给定一个类别y和一个从x1到xn的相关的特征向量。
        假设每对特征之间相互独立；尽管假设简单，但很多实际情况下，朴素贝叶斯效果很好，如文档分类和垃圾邮件过滤。
        相比于其他复杂方法，朴素贝叶斯学习器和分类器非常快。
        分类条件分布的解耦意味着可以独立单独地把每个特征视为1维分布来估计，反过来有助于缓解维度灾难问题。
        是一种不错的分类器，但不是好的估计器（estimator），所以不能太过于重视predict_proba输出的概率。
        
        使用最大后验概率（MAP）来估计P(y)【训练集中类别y的相对概率】和P(x_i|y)
        各种朴素贝叶斯分类器的差异主要来自于处理P(x_i|y)分布时所做的假设不同。
    1.9.1 高斯朴素贝叶斯
        GaussianNB实现了用于分类的高斯朴素贝叶斯算法。
        特征的可能性(即概率)假设为高斯分布；
    1.9.2 多项分布朴素贝叶斯
        MultinomialNB实现，也是用于文本分类(这个领域中数据往往以词向量表示，尽管在实践中 tf-idf 向量在预测时表现良好)的两大经典朴素贝叶斯算法之一。
        分布参数由每类y的θ向量决定。
    1.9.3. 伯努利朴素贝叶斯
        BernoulliNB实现了用于多重伯努利分布数据的朴素贝叶斯训练和分类算法；
        有多个特征，但每个特征都假设是一个二元(Bernoulli)变量。算法要求样本以二元值特征向量表示，如果有其他类型，BernoulliNB实例会将其二值化。
    1.9.4. 堆外朴素贝叶斯模型拟合
        解决整个训练集不能导入内存的大规模分类问题。
        MultinomialNB, BernoulliNB, 和 GaussianNB 实现了 partial_fit 方法，可以动态的增加数据，使用方法与其他分类器的一样；
        与fit不同，首次调用partial_fit方法需要传递一个所有期望的类标签的列表。
"""

"""
1.10 决策树
    Decision Trees是一种classification和regression的无参监督学习方法，创建模型从特征中选择简单的决策规则来预测一个目标变量的值。
    1.10.1 分类
        DecisionTreeClassifier类实现，输入两个数组：X[n_samples, n_features]存放数组，Y[n_samples]存放标签。
        fit() predict() predict_proba()返回概率；
        可以使用 export_graphviz 导出器以 Graphviz 格式导出决策树；
    1.10.2 回归
        DecisionTreeRegressor类实现，
    1.10.4 复杂度分析
        构
    1.10.5 技巧
        > * 对于拥有大量特征的数据决策树会出现过拟合的现象。获得一个合适的样本比例和特征数量十分重要，因为在高维空间中只有少量的样本的树是十分容易过拟合的。 > * 考虑事先进行降维( PCA , ICA ，使您的树更好地找到具有分辨性的特征。 > * 通过 export 功能可以可视化您的决策树。使用 max_depth=3 作为初始树深度，让决策树知道如何适应您的数据，然后再增加树的深度。 > * 请记住，填充树的样本数量会增加树的每个附加级别。使用 max_depth 来控制输的大小防止过拟合。 > * 通过使用 min_samples_split 和 min_samples_leaf 来控制叶节点上的样本数量。当这个值很小时意味着生成的决策树将会过拟合，然而当这个值很大时将会不利于决策树的对样本的学习。所以尝试 min_samples_leaf=5 作为初始值。如果样本的变化量很大，可以使用浮点数作为这两个参数中的百分比。两者之间的主要区别在于 min_samples_leaf 保证叶结点中最少的采样数，而 min_samples_split 可以创建任意小的叶子，尽管在文献中 min_samples_split 更常见。 > * 在训练之前平衡您的数据集，以防止决策树偏向于主导类.可以通过从每个类中抽取相等数量的样本来进行类平衡，或者优选地通过将每个类的样本权重 (sample_weight) 的和归一化为相同的值。还要注意的是，基于权重的预修剪标准 (min_weight_fraction_leaf) 对于显性类别的偏倚偏小，而不是不了解样本权重的标准，如 min_samples_leaf 。
        如果样本被加权，则使用基于权重的预修剪标准 min_weight_fraction_leaf 来优化树结构将更容易，这确保叶节点包含样本权重的总和的至少一部分。
        所有的决策树内部使用 np.float32 数组 ，如果训练数据不是这种格式，将会复制数据集。
        如果输入的矩阵X为稀疏矩阵，建议您在调用fit之前将矩阵X转换为稀疏的csc_matrix ,在调用predict之前将 csr_matrix 稀疏。当特征在大多数样本中具有零值时，与密集矩阵相比，稀疏矩阵输入的训练时间可以快几个数量级。

1.11 集成方法
    


"""