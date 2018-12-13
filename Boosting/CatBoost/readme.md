# CatBoost介绍

**CatBoost**是俄罗斯的搜索巨头Yandex在2017年开源的机器学习库，是**Gradient Boosting**(**梯度提升**) + **Categorical Features**(**类别型特征**)。类似于LightGBM，也是基于梯度提升决策树的机器学习框架。[关于CatBoost参见](https://tech.yandex.com/catboost/)，[CatBoost算法参见](http://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf)

### 一，CatBoost程序文件
### 二，CatBoost的优点(官宣)

1. 性能卓越：在性能方面可以匹敌任何先进的机器学习算法；

1. 鲁棒性/强健性：它减少了对很多超参数调优的需求，并降低了过度拟合的机会，这也使得模型变得更加具有通用性。

1. 易于使用：提供与scikit集成的Python接口，以及R和命令行界面；

1. 实用：可以处理类别型、数值型特征；

1. 可扩展：支持自定义损失函数；

### 三，CatBoost介绍

以下内容主要翻译自[论文](http://learningsys.org/nips17/assets/papers/paper_11.pdf)


### 论文题目：

**CatBoost: gradient boosting with categorical features support**

  CatBoost: 支持类别型特征梯度提升的库

##### 作者：

**Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin**

##### 摘要：

本文介绍了一个新的开源的梯度提升库——CatBoost，它可以很好地处理类别型特征，并且在许多流行的公共数据集上，其性能超越了目前同样基于梯度提升的其他算法。这个库学习基于GPU实现，评分基于CPU实现，针对许多不同尺寸的数据样本集合，其速度都明显快于其他梯度提升库。

### 1 引言

梯度提升是一种强大的机器学习算法，在许多不同领域的应用中都能获得非常好的结果。多年来，在解决异构、噪声和具有复杂依赖关系的数据的学习问题时，例如web搜索，推荐系统、天气预报以及其他方面的问题，它始终是首选方案。通过在函数空间内利用梯度下降，并且在这个过程中利用贪婪策略建模的理论，可以说明通过迭代组合模型(弱预测器)来构建强预测器是可行的。

大多数流行的梯度提升算法利用决策树作为基本预测器。对于数值型特征使用决策树很方便，但是实际中，许多数据集包括类别型特征，这些特征对预测也很重要。类别型特征具有离散的值，并且对这些值进行比较(例如用户ID，城市名称)意义不大。梯度提升算法中处理这类特征的最常用的方法就是在学习之前，也就是数据预处理阶段，将这些特征的值转换为数字。

本文提出了一种新的可以很好的处理类别型特征的梯度提升算法，该算法的改进之处就在于在学习的时候处理这些特征，而不是在数据预处理阶段。该算法的另一个优点是使用新的方法计算叶子节点的值来生成树，并且这种计算方式有助于减少过拟合。

在许多不同的流行数据集上的，本文算法的性能均优于目前先进的梯度提升算法库GBDT，XGBoost，LightGBM以及H2O。这个算法名为CatBoost（“分类提升”），源码已经开源。

CatBoost支持CPU和GPU实现。GPU实现使得学习更快，并且在很多不同大小的数据集上，速度都快于开源的GBDT的GPU实现，以及XGBoost和LightGBM。该库可实现快速的CPU评分，经过验证，其性能超过XGBoost和LightGBM。

### 2 类别型特征

类别型特征，也就是其值(可称为类别)是离散的特征，并且值之间的比较意义不大的，因此这样的特征不能在二叉树中直接使用。通常的做法就是在数据预处理阶段将他们转化为数字。例如将样本中类别型特征的值转换为1个或者多个数字。
 
对于类别数较少的类别型特征，常用的处理方式就是独热编码，也就是将原始的特征去除掉，然后将转码后的特征加到数据中。在数据预处理阶段和学习阶段都可以执行独热编码。而CatBoost中实现了在学习阶段执行的一种更为有效的方式。

另一种处理类别型特征的方式就是利用样本的标签计算一些统计。对于给定的数据集

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\textit{D}=\{(\mathbf{X1},\mathbf{Y1}),(\mathbf{X2},\mathbf{Y2}),\cdots&space;,(\mathbf{Xn},\mathbf{Yn})\}\:&space;where\,&space;\mathbf{Xi}=(xi1,&space;xi2,&space;\cdots&space;xim)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\textit{D}=\{(\mathbf{X1},\mathbf{Y1}),(\mathbf{X2},\mathbf{Y2}),\cdots&space;,(\mathbf{Xn},\mathbf{Yn})\}\:&space;where\,&space;\mathbf{Xi}=(xi1,&space;xi2,&space;\cdots&space;xim)" title="\\ \textit{D}=\{(\mathbf{X1},\mathbf{Y1}),(\mathbf{X2},\mathbf{Y2}),\cdots ,(\mathbf{Xn},\mathbf{Yn})\}\: where\, \mathbf{Xi}=(xi1, xi2, \cdots xim)" /></a>

n为数据及样本的个数，m为特征的个数。其中一些特征是数值型的，令一些是类别型的。**Yi**表示数据样本的标签值。最简单的方式就是用整个数据集的平均标签值代替类别特征的值。也就是对于样本i的特征k的值应该变为：

<a href="https://www.codecogs.com/eqnedit.php?latex=xik=\frac{\sum_{j=1}^{n}[xjk=xik]\cdot&space;Yj}{\sum_{j=1}^{n}[xjk=xik]}&space;\:&space;where&space;\;&space;\:&space;[a=b]=\left\{\begin{matrix}&space;1\,&space;\,&space;\,&space;a=b\\&space;0\,&space;\,&space;\,&space;a\neq&space;b&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?xik=\frac{\sum_{j=1}^{n}[xjk=xik]\cdot&space;Yj}{\sum_{j=1}^{n}[xjk=xik]}&space;\:&space;where&space;\;&space;\:&space;[a=b]=\left\{\begin{matrix}&space;1\,&space;\,&space;\,&space;a=b\\&space;0\,&space;\,&space;\,&space;a\neq&space;b&space;\end{matrix}\right." title="xik=\frac{\sum_{j=1}^{n}[xjk=xik]\cdot Yj}{\sum_{j=1}^{n}[xjk=xik]} \: where \; \: [a=b]=\left\{\begin{matrix} 1\, \, \, a=b\\ 0\, \, \, a\neq b \end{matrix}\right." /></a>

很明显，这样做会导致过拟合。例如，某个类别型特征中，如果某个类别值只有一个样本，那么该值转换后的数值就等于这个样本的标签值(也就是上式中的分母为1，分子为标签值\*1，因此相除的结果就是标签值)。避免此问题的一个直接的做法就是将数据集划分为2部分，其中一部分用于计算用于替换的值，另一部分用来学习。这种方式虽然降低了过拟合，但是同时也减少了学习、计算时的样本数量。

CatBoost采用了一种更为有效的策略，降低过拟合的同时也保证了全部数据集都可用于学习。也就是对数据集进行随机排列，计算相同类别值的样本的平均标签值时，只是将这个样本之前的样本的标签值纳入计算。假设<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma&space;=(\sigma_1,\sigma_2,\cdots&space;\sigma&space;_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma&space;=(\sigma_1,\sigma_2,\cdots&space;\sigma&space;_n)" title="\sigma =(\sigma_1,\sigma_2,\cdots \sigma _n)" /></a>为一个排列，则有下式：

<a href="https://www.codecogs.com/eqnedit.php?latex=xik=\frac{\sum_{j=1}^{n}[xjk=xik]\cdot&space;Yj}{\sum_{j=1}^{n}[xjk=xik]}&space;\:&space;where&space;\;&space;\:&space;[a=b]=\left\{\begin{matrix}&space;1\,&space;\,&space;\,&space;a=b\\&space;0\,&space;\,&space;\,&space;a\neq&space;b&space;\end{matrix}\right.&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?xik=\frac{\sum_{j=1}^{n}[xjk=xik]\cdot&space;Yj}{\sum_{j=1}^{n}[xjk=xik]}&space;\:&space;where&space;\;&space;\:&space;[a=b]=\left\{\begin{matrix}&space;1\,&space;\,&space;\,&space;a=b\\&space;0\,&space;\,&space;\,&space;a\neq&space;b&space;\end{matrix}\right.&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;(1)" title="xik=\frac{\sum_{j=1}^{n}[xjk=xik]\cdot Yj}{\sum_{j=1}^{n}[xjk=xik]} \: where \; \: [a=b]=\left\{\begin{matrix} 1\, \, \, a=b\\ 0\, \, \, a\neq b \end{matrix}\right. \, \, \, \, \, \, \, (1)" /></a>

其中P是先验项，先验项的权重a>0。添加先验项是一个普遍做法，针对类别数较少的特征，它可以减少噪音数据。对于回归问题，一般情况下，先验项可取数据集的均值。对于二分类，先验项是正例的先验概率。利用多个数据集排列也是有效的，但是，如果直接计算可能导致过拟合。CatBoost利用了一个比较新颖的计算叶子节点值的方法，这种方式可以避免多个数据集排列中直接计算会出现过拟合的问题，这将在下一节中讨论，

#### 2.1 特征组合

值得注意的是几个类别型特征的任意组合都可视为新的特征。例如，在音乐推荐应用中，我们有两个类别型特征：用户ID和音乐流派。如果有些用户更喜欢摇滚乐，将用户ID和音乐流派转换为数字特征时，根据（公式1）这些信息就会丢失。结合这两个特征就可以解决这个问题，并且可以得到一个新的强大的特征。然而，组合的数量会随着数据集中类别型特征的数量成指数增长，因此不可能在算法中考虑所有组合。为当前树构造新的分割点时，CatBoost会采用贪婪的策略考虑组合。对于树的第一次分割，不考虑任何组合。对于下一个分割，CatBoost将当前树的所有组合、类别型特征与数据集中的所有类别型特征相结合。组合被动态地转换为数字。CatBoost还通过以下方式生成数值型特征和类别型特征的组合：树选择的所有分割点都被视为具有两个值的类别型特征，并且组合方式和类别型特征一样。


#### 2.2 重要的实现细节

用数字代替类别值的另一种方法是计算该类别值在数据集特征中出现的次数。这是一种简单但强大的技术，在CatBoost也有实现。这种统计量也适用于特征组合。

CatBoost算法为了在每个步骤中拟合最优的先验条件，我们考虑多个先验条件，为每个先验条件构造一个特征，这在质量上比提到的标准技术更有效。

### 3 克服梯度偏差

CatBoost，和所有标准梯度提升算法一样，都是通过构建新树来拟合当前模型的梯度。然而，所有经典的提升算法都存在由有偏的点态梯度估计引起的过拟合问题。在每个步骤中使用的梯度都使用当前模型中的相同的数据点来估计，这导致估计梯度在特征空间的任何域中的分布与该域中梯度的真实分布相比发生了偏移，从而导致过拟合。关于有偏梯度的概念在文献[1]，[9]中已经讨论过。我们在文[5]对这个问题进行了详细的分析，为了解决这一问题，还对经典的梯度提升算法进行了许多改进。CatBoost实现了这些改进之一，下面简要介绍：

在许多利用GBDT技术的算法（例如，XGBoost、LightGBM）中，构建下一棵树分为两个阶段：选择树结构和在树结构固定后计算叶子节点的值。为了选择最佳的树结构，算法通过枚举不同的分割，用这些分割构建树，对得到的叶子节点中计算值，然后对得到的树计算评分，最后选择最佳的分割。两个阶段叶子节点的值都是被当做梯度[8]或牛顿步长的近似值来计算。在CatBoost中，第二阶段使用传统的GBDT方案执行，第一阶段使用修改后的版本。

根据我们在文[5]中的经验结果和理论分析可知，使用梯度步长的无偏估计是很有必要的。设<a href="https://www.codecogs.com/eqnedit.php?latex=F^{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^{i}" title="F^{i}" /></a>为构建i棵树后的模型，<a href="https://www.codecogs.com/eqnedit.php?latex=g^{i}(\mathbf{X_{k}},Y_{k})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g^{i}(\mathbf{X_{k}},Y_{k})" title="g^{i}(\mathbf{X_{k}},Y_{k})" /></a>为建立i棵树后第k个训练样本上的梯度值。为了使梯度<a href="https://www.codecogs.com/eqnedit.php?latex=g^{i}(\mathbf{X_{k}},Y_{k})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g^{i}(\mathbf{X_{k}},Y_{k})" title="g^{i}(\mathbf{X_{k}},Y_{k})" /></a>无偏于模型<a href="https://www.codecogs.com/eqnedit.php?latex=F^{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^{i}" title="F^{i}" /></a>，我们需要在没有<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{X_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{X_{k}}" title="\mathbf{X_{k}}" /></a>的情况下对<a href="https://www.codecogs.com/eqnedit.php?latex=F^{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^{i}" title="F^{i}" /></a>进行训练。由于我们需要对所有训练样本计算无偏的梯度估计，所以不能使用任何值来训练<a href="https://www.codecogs.com/eqnedit.php?latex=F^{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^{i}" title="F^{i}" /></a>，乍一看这不可能实现。我们运用以下技巧来处理这个问题：对于每个示例<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{X_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{X_{k}}" title="\mathbf{X_{k}}" /></a>，我们训练一个单独的模型<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{k}}" title="\mathbf{M_{k}}" /></a>，该模型<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{k}}" title="\mathbf{M_{k}}" /></a>从未使用此示例的梯度估计进行更新。使用<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{k}}" title="\mathbf{M_{k}}" /></a>，我们估计<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{X_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{X_{k}}" title="\mathbf{X_{k}}" /></a>上的梯度，并使用这个估计对结果树进行评分。下面通过伪代码说明此技巧。设**Loss**(y，a)为损失函数，其中y为标签值，a为公式值。

值得注意的是<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{i}}" title="\mathbf{M_{i}}" /></a>模型的建立没有样本<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{X_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{X_{i}}" title="\mathbf{X_{i}}" /></a> 的参与。CatBoost中所有的树<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{i}}" title="\mathbf{M_{i}}" /></a>的结构都是一样的。

在CatBoost中，我们生成训练数据集的s个随机排列。采用多个随机排列是为了增强算法的鲁棒性：针对每一个随机排列，计算得到其梯度。这些排列与用于计算类别型特征的统计量时的排列相同。我们使用不同的排列来训练不同的模型，因此不会导致过拟合。对于每个排列，我们训练n个不同的模型<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{i}}" title="\mathbf{M_{i}}" /></a>，如上所示。这意味着为了构建一棵树，需要对每个排列存储并重新计算，其时间复杂度近似于O(n^2)：对于每个模型<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{i}}" title="\mathbf{M_{i}}" /></a>，我们必须更新<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{1}}(\mathbf{X_{1}}),\cdots,&space;\mathbf{M_{i}}(\mathbf{X_{i}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{1}}(\mathbf{X_{1}}),\cdots,&space;\mathbf{M_{i}}(\mathbf{X_{i}})" title="\mathbf{M_{1}}(\mathbf{X_{1}}),\cdots, \mathbf{M_{i}}(\mathbf{X_{i}})" /></a>。因此，时间复杂度变成O(sn^2)。在我们的实现中，我们使用一个重要的技巧，可以将构建一个树的时间复杂度降低到O(sn)：对于每个排列，我们不执行时间复杂度为O(n^2)的存储和更新<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{M_{i}}(\mathbf{X_{j}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{M_{i}}(\mathbf{X_{j}})" title="\mathbf{M_{i}}(\mathbf{X_{j}})" /></a>值的操作，而是保持值<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{M}'_{i}}(\mathbf{X_{j}}),i=1,2,\cdots&space;[log_{2}n],j<2^{i&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{M}'_{i}}(\mathbf{X_{j}}),i=1,2,\cdots&space;[log_{2}n],j<2^{i&plus;1}" title="\mathbf{{M}'_{i}}(\mathbf{X_{j}}),i=1,2,\cdots [log_{2}n],j<2^{i+1}" /></a>，其中<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{M}'_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{M}'_{i}}" title="\mathbf{{M}'_{i}}" /></a>是样本j前的2^i样本的近似值。因此，预测值<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{M}'_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{M}'_{i}}" title="\mathbf{{M}'_{i}}" /></a>不大于<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{0\leq&space;i&space;\leq&space;log_{2}n&space;}2^{i&plus;1}<&space;4n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{0\leq&space;i&space;\leq&space;log_{2}n&space;}2^{i&plus;1}<&space;4n" title="\sum_{0\leq i \leq log_{2}n }2^{i+1}< 4n" /></a>，用于选择树结构的样本Xk上的梯度就近似等于<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{M}'_{k}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{M}'_{k}}" title="\mathbf{{M}'_{k}}" /></a>，其中i=[log2(k)]。

### 4 快速评分

CatBoost使用oblivious树作为基本预测器。在这类树中，相同的分割准则在树的整个级别上使用[12，13]。这种树是平衡的，不太容易过拟合。梯度提升oblivious树被成功地用于各种学习任务[7，10]。在oblivious树中，每个叶子节点的索引可以被编码为长度等于树深度的二进制向量。这在CatBoost模型评估器中得到了广泛的应用：我们首先将所有浮点特征、统计信息和独热编码特征进行二值化，然后使用二进制特征来计算模型预测值。

所有样本的所有二进制特征值都存储在连续向量B中。叶子节点的值存储在大小为2^d的浮点数向量中，其中d是树的深度。为了计算第t棵树的叶子节点的索引，对于样本x，我们建立了一个二进制向量<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=0}^{d-1}2^{i}\cdot&space;B(x,f(t,j))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=0}^{d-1}2^{i}\cdot&space;B(x,f(t,j))" title="\sum_{i=0}^{d-1}2^{i}\cdot B(x,f(t,j))" /></a>，其中<a href="https://www.codecogs.com/eqnedit.php?latex=B(x,f)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B(x,f)" title="B(x,f)" /></a>是从向量B读取的样本x上的二进制特征f的值，而<a href="https://www.codecogs.com/eqnedit.php?latex=f(t,i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(t,i)" title="f(t,i)" /></a>是从深度i上的第t棵树中的二进制特征的数目。

这些向量可以以数据并行的方式构建，这种方式可以实现高达3倍的加速。正如我们的实验所示，这比所有现有的评分器快得多，。

### 5 基于GPU实现快速学习

#### 5.1 密集的数值特征

对于任何GBDT算法而言，最大的难点之一就是搜索最佳分割。尤其是对于密集的数值特征数据集来说，该步骤是建立决策树时的主要计算负担。CatBoost使oblivious 决策树作为基础模型，并将特征离散化到固定数量的箱子中以减少内存使用[10]。箱子的数量是算法的参数。因此，我们可以使用基于直方图的方法来搜索最佳分割。我们在GPU上构建决策树的方法在本质上类似于[11]中所描述的方法。我们利用一个32位整数将多个数值型特征打包，规则：

  * **存储二进制特征用1位，每个整数包括32个特征**
  * **存储不超过15个值的特征用4位，每个整数包括8个特征**
  * **存储其他特征用8位（不同值的个数最大是255），每个整数包括4个特征**

就GPU内存使用而言，CatBoost至少与LightGBM[11]一样有效。主要不同是利用一种不同的直方图计算方法。LightGBM和XGBoost4的算法有一个主要缺点：它们依赖于原子操作。这种技术在内存上很容易处理，但是在性能好的GPU上，它会比较慢。事实上直方图可以在不涉及原子操作的情况下更有效地计算。仅通过一个简化例子说明我们方法的基本思想：同时计算四个32箱的直方图，每个特征都有一个浮点统计计算。对于具有多个统计量和多个直方图的情况，这种思想都适用。

现在有梯度值g[i]和特征组<a href="https://www.codecogs.com/eqnedit.php?latex=(f_1,f_2,f_3,f_4)[i]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(f_1,f_2,f_3,f_4)[i]" title="(f_1,f_2,f_3,f_4)[i]" /></a>。因此需要计算4个直方图<a href="https://www.codecogs.com/eqnedit.php?latex=\mathrm{hist}[j][b]&space;=&space;\sum&space;_{i:f_j[i]=b}g[i]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathrm{hist}[j][b]&space;=&space;\sum&space;_{i:f_j[i]=b}g[i]" title="\mathrm{hist}[j][b] = \sum _{i:f_j[i]=b}g[i]" /></a>。CatBoost为每个warp直方图构建部分直方图，而不是为每个线程块构建直方图。下面我们将描述一个warp在前32个样本上完成的工作。索引为i的线程处理样本i。由于我们同时构建了4个直方图，因此每warp需要32\*32\*4字节的共享内存。为了更新直方图，32个线程都将样本标签和分组特征加载到寄存器。然后，warp在4次迭代中同时执行共享内存直方图的更新：在第l次(l=0，1，2，3)迭代时，索引为i的线程处理特征<a href="https://www.codecogs.com/eqnedit.php?latex=f_{(l&plus;i)&space;mod&space;4}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{(l&plus;i)&space;mod&space;4}" title="f_{(l+i) mod 4}" /></a>，并且将g[i]加到<a href="https://www.codecogs.com/eqnedit.php?latex=\mathrm{hist}[(l&space;&plus;&space;i)\:&space;mod&space;\,&space;4][f_{(l&plus;i)&space;\,&space;mod&space;\:&space;4}]." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathrm{hist}[(l&space;&plus;&space;i)\:&space;mod&space;\,&space;4][f_{(l&plus;i)&space;\,&space;mod&space;\:&space;4}]." title="\mathrm{hist}[(l + i)\: mod \, 4][f_{(l+i) \, mod \: 4}]." /></a>]。通过适当的直方图布局，可以避免冲突，所有32个线程并行地添加统计数据。

CatBoost构建直方图时，每个组8个特征和2个统计计算；32个二进制特征和2个统计量；4个特征和2个统计量，箱子数可以为32、64、128和255。为了实现所有这些直方图的快速计算，必须利用所有可用的共享内存。但是我们的代码无法实现100%的占用率。所以我们利用指令级并行性进行循环展开。这种技术甚至在较低的占用率下也能实现高性能。

#### 5.2 类别型特征

CatBoost实现了多种处理类别型特征的方法。对于独热编码特征，我们不需要任何特殊处理，基于直方图的分割搜索方法可以很容易地处理这种情况。在数据预处理阶段，就可以对单个类别型特征进行统计计算。CatBoost还对特征组合使用统计信息。处理它们是算法中速度最慢、消耗内存最多的部分。

我们使用完美哈希来存储类别特性的值，以减少内存使用。由于GPU内存的限制，我们在CPU RAM中存储按位压缩的完美哈希，以及要求的数据流、重叠计算和内存等操作。动态地构造特征组合要求我们为这个新特征动态地构建（完美）哈希函数，并为哈希的每个唯一值计算关于某些排列的统计数据。我们使用基数排序来构建完美的哈希，并通过哈希来分组观察。在每个组中，我们需要计算一些统计量的前缀和。该统计量的计算使用分段扫描GPU图元进行（CatBoost分段扫描实现通过算子变换[16]完成，并且基于CUB[6]中扫描图元的高效实现）。

#### 5.3 多GPU支持

CatBoost中的GPU实现可支持多个GPU。分布式树学习可以通过样本或特征进行并行化。CatBoost使用具有多个学习数据集排列的计算方案，并在训练期间计算分类特征的统计数据。因此，我们需要利用特征并行学习。



