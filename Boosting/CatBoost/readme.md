# CatBoost介绍

**CatBoost**是俄罗斯的搜索巨头Yandex在2017年开源的机器学习库，是**Gradient Boosting**(**梯度提升**) + **Categorical Features**(**类别型特征**)。类似于LightGBM，也是基于梯度提升决策树的机器学习框架。[详情参见](https://tech.yandex.com/catboost/)。

### 1，[CatBoost介绍]

以下内容主要翻译自[论文](http://learningsys.org/nips17/assets/papers/paper_11.pdf)


### 论文题目：

**CatBoost: gradient boosting with categorical features support**

##### 作者：

**Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin**

##### 摘要：



#### 1.1 类别型特征

   类别型特征，也就是具有离散值的特征，例如类似收入(高，低)、成绩(优秀，良好，差)、天气(晴天，阴天)这样用类别值表示的特征。可以通过数字编码来处理这类特征，也就是每个类别值对应一个数字。其中独热编码是广泛应用的一种处理方式，可以在数据预处理阶段或者训练阶段使用，CatBoost就是在训练阶段才使用的。


#### 2，CatBoost的优点(官宣)

1. 性能卓越：在性能方面可以匹敌任何先进的机器学习算法；

1. 鲁棒性/强健性：它减少了对很多超参数调优的需求，并降低了过度拟合的机会，这也使得模型变得更加具有通用性。

1. 易于使用：提供与scikit集成的Python接口，以及R和命令行界面；

1. 实用：特征值可以为字符串或者数字，无需将字符串经过编码；

1. 可扩展：支持自定义损失函数；


### 2，CatBoost程序文件
