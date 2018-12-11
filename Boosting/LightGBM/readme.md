# LightGBM介绍

**LightGBM**(**Light Gradient Boosting Machine**)是微软的开源分布式高性能Gradient Boosting框架，使用基于决策树的学习算法。

### LightGBM涉及到的优化([官方文档翻译](https://github.com/Microsoft/LightGBM/blob/master/docs/Features.rst))

* **速度、内存方面的优化**

    许多提升工具使用基于预排序的算法(近似直方图算法)（例如XGBoost中的默认算法）来进行决策树学习。这是一个比较简单的解决方案，但不容易优化。LightGBM使用基于直方图的算法，它将连续特征值存储到离散区间。这可以加快训练速度并减少了内存使用量。

   * **直方图算法的优点**

      + **降低计算分裂增益的成本**
      
        1. 基于预排序的算法具有时间复杂性 **O**(训练样本的个数)，计算直方图具有时间复杂度**O**(训练样本的个数)，但这仅当在执行总结操作时发生；
      
        2. 构建完直方图后，基于直方图的算法具有时间复杂度**O**(某个特征不同值的个数)，因为某个特征不同值的个数远小于训练样本的个数；
      
      + **使用直方图的相减进一步加速**
      
         1. 在计算某个叶子节点的直方图时，通过它的父节点和它的相邻节点的直方图相减，得到；
         
         2. 根据上条可知，每次分裂仅需要为一个叶子节点构建直方图，另一个叶子节点的可以通过直方图的减法获得，成本较低；
      
      + **减少内存的使用**
      
           用离散箱替换连续值。如果某个特征不同值的个数很小，可以使用小数据类型，例如uint8_t来存储训练数据，无需存储用于预排序特征值的其他信息
       
      + **降低并行学习的通信成本**
      
* **针对稀疏特征优化**

     对于稀疏的特征只需要O(2 * 非零值的样本个数)的时间复杂度来构造直方图
     
* **优化树的生长策略来提高准确率**

    + **最好的叶子节点优先分裂(Leaf_wise)的决策树生成策略**
     
      大多数决策树学习算法生成树的策略是以同样的深度生成树(Level_wise)，生成树的示例见下图：

   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/level_wise.png)

     而LightGBM以Leaf_wise的方式生成。它将选择具有最大增益损失的叶子节点来分裂。当两种方式生成的树具有相等的叶子节点时，Leaf_wise策略生成的树会比Level_wise的拟合度更高。

    对于样本较少的情况，Leaf_wise方式可能会导致过度拟合，因此LightGBM中利用参数max_depth来限制树的深度。Leaf_wise方式生成的树示例如下：

   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/leaf_wise.png)

   + **特征的最优分割点**
  
       通常使用独热编码来转换分类特征，但这种方法对于决策树的学习并不是有益的。特别是对于不同值较多的特征，独热编码后构建的树往往是不平衡的，并且需要非常大的深度才能获得较好的准确率。

       事实上，最佳解决方案是通过将类别划分为2个子集。如果特征具有K个不同的值，就是在**2^(k-1) - 1**种情况里找到最优的分割点。对于回归树而言，有一种解决方案可以保证在**O**(k * log(k))的时间复杂度内找到最优分割点。

      找到特征的最优分割点的基本思想是根据训练目标的相关性对类别进行重排序。 更具体的说，根据累加值(sum_gradient / sum_hessian)重新对（类别特征的）直方图进行排序，然后在排好序的直方图中寻找最优的分割点。

* **网络通信的优化**

     在LightGBM的并行学习中，它只需要使用一些聚合通信算法，如“All reduce”，“All gather”和“Reduce scatter”。LightGBM实现了最先进的state-of-art算法。这些聚合通信算法可以提供比点对点通信更好的性能。

* **并行学习中的优化**

    LightGBM提供以下并行学习算法的优化：特征并行、数据并行、投票并行。

   * **特征并行**
   
     传统算法中的特征并行，主要是体现在找到最好的分割点，其步骤为：

     1. 垂直分区数据（不同的线程具有不同的数据集）；
     2. 在本地数据集上找到最佳分割点，包括特征，阈值；
     2. 再进行各个划分的通信整合并得到最佳划分；
     2. 以最佳划分方法对数据进行划分，并将数据划分结果传递给其他线程；
     2. 其他线程对接受到的数据进一步划分；

      传统特征并行的缺点：

      1. 计算成本较大，传统特征并行没有实现得到"split"（时间复杂度为“O（训练样本的个数)"）的加速。当数据量很大的时候，难以加速；
      1. 需要对划分的结果进行通信整合，其额外的时间复杂度约为 “O（训练样本的个数/8）”（一个数据一个字节）；
      
      #### LightGBM中的并行特征
      
      由于特征并行在训练样本的个数大的时候不能很好地加速，LightGBM做了以下优化：不是垂直分割数据，而是每个线程都拥有完整的全部数据。因此，LightGBM不需要为分割数据结果进行通信，因为每个线程都知道如何划分数据。并且训练样本的个数不会变大，因此这种方案是合理的。

      LightGBM中实现特征并行的过程：

      1. 每个线程在本地数据集上找到最佳分割点，包括特征，阈值；
      1. 本地进行各个划分的通信整合并得到最佳划分；
      1. 执行最佳划分；
      
      然而，该并行算法在数据量很大时仍然存在计算上的局限。因此，建议在数据量很大时使用数据并行。

   * **数据并行**
   
      传统算法数据并行旨在并行化整个决策学习。数据并行的过程是：

     1. 水平划分数据；
     1. 线程以本地数据构建本地直方图；
     1. 将本地直方图整合成全局直方图；
     1. 在全局直方图中寻找最佳划分，然后执行此划分；
     
      传统数据并行的缺点：通信成本高。如果使用点对点通信算法，则一台机器的通信成本约为O(#machine * #feature * #bin)。如果使用聚合通信算法（例如“All Reduce”），通信成本约为O(2 * #feature * #bin)。
            
      #### LightGBM中的数据并行
      
      LightGBM中通过下面方法来降低数据并行的通信成本：

      1. 不同于“整合所有本地直方图以形成全局直方图”的方式，LightGBM 使用分散规约(Reduce scatter)的方式对不同线程的不同特征（不重叠的）进行整合。 然后线程从本地整合直方图中寻找最佳划分并同步到全局的最佳划分中；
      1. LightGBM通过直方图的减法加速训练。 基于此，我们可以进行单叶子节点的直方图通讯，并且在相邻直方图上作减法；
      
      通过上述方法，LightGBM 将数据并行中的通讯开销减少到O(0.5 * #feature * #bin)。

  * **投票并行**
  
     投票并行进一步降低了数据并行中的通信成本，使其减少至常数级别。它使用两阶段投票来降低特征直方图的通信成本。

* **GPU支持**

* **支持的应用和度量**

     ##### LightGBM支持以下应用：

   * **回归，目标函数是L2损失**
   * **二进制分类，目标函数是logloss(对数损失)**
   * **多分类**
   * **交叉熵，目标函数是logloss，支持非二进制标签的训练**
   * **lambdarank，目标函数为基于NDCG的lambdarank**
   
     ##### LightGBM支持以下度量：

  * **L1 loss：绝对值损失**
  * **L2 loss：MSE，平方损失**
  * **Log loss：对数损失**
  * **分类错误率**
  * **AUC（Area Under Curve）：ROC曲线下的面积**
  * **NDCG（Normalized Discounted Cumulative Gain）：归一化折损累积增益**
  * **MAP（Mean Average Precision）：平均精度均值**
  * **多类别对数损失**
  * **多类别分类错误率**
  * **Fair损失**
  * **Huber损失**
  * **Possion：泊松回归**
  * **Quantile：分位数回归**
  * **MAPE（Mean Absolute Percent Error）：平均绝对误差百分比**
  * **kullback_leibler：Kullback-Leibler divergence**
  * **gamma：negative log-likelihood for Gamma regression**
  * **tweedie, negative log-likelihood for Tweedie regression**
  
      [更多详情点击](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#metric-parameters)

     ##### 其他特性
      
  * **限制树的最大深度max_depth**
  * **DART**
  * **L1 / L2正则化**
  * **套袋**
  * **随即选择列(特征)子集**
  * **Continued train with input GBDT model**
  * **Continued train with the input score file**
  * **Weighted training**
  * **Validation metric output during training**
  * **交叉验证**
  * **Multi metrics**
  * **提前停止（训练和预测）**
  * **叶指数的预测**
  
       [更多详情点击](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst) 
