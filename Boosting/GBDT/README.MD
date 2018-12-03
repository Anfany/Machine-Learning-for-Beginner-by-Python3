# GBDT


* **GBDT程序文件**

* **GBDT说明**

   **GBDT**是Gradient Boosting Decison Tree的简称，其中Gradient是梯度，是这个方法的核心；Boosting是提升，是这个方法的框架；Decision Tree是决策树，是实现这个方法用到的模型。理解了梯度，也就理解了GBDT。

   GBDT可以解决回归问题，经过一些处理也可以解决分类问题，但是用到的树都是回归树，这一点需要牢记。
  
   下面通过简单的回归例子，说明此方法的大致流程：
   
   给定一个训练数据集合<a href="https://www.codecogs.com/eqnedit.php?latex=Data\_Train&space;=&space;\{(X1,Y1),&space;(X2,Y2),\cdots&space;,(Xn,Yn)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data\_Train&space;=&space;\{(X1,Y1),&space;(X2,Y2),\cdots&space;,(Xn,Yn)\}" title="Data\_Train = \{(X1,Y1), (X2,Y2),\cdots ,(Xn,Yn)\}" /></a>，其中输出值的序列为**P_T0**；
   
   首先建立一个CART回归树**T1**，训练数据集的预测输出序列为**P_T1**(树T1叶子节点的平均值)。这里我们用MSE作为这个树T1的损失函数**C_T1**：
   
   <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{C\_T1}=\frac{1}{2}\sum_{i=1}^{n}(P\_T0\_i&space;-&space;P\_T1\_i)^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{C\_T1}=\frac{1}{2}\sum_{i=1}^{n}(P\_T0\_i&space;-&space;P\_T1\_i)^{2}" title="\mathbf{C\_T1}=\frac{1}{2}\sum_{i=1}^{n}(P\_T0\_i - P\_T1\_i)^{2}" /></a>
   
   此时计算损失函数对于**P_T0**的梯度：
   
   <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\mathbf{C\_T1}&space;}{\partial&space;P\_T0\_i}&space;=&space;P\_T0\_i&space;-&space;P\_T1\_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\mathbf{C\_T1}&space;}{\partial&space;P\_T0\_i}&space;=&space;P\_T0\_i&space;-&space;P\_T1\_i" title="\frac{\partial \mathbf{C\_T1} }{\partial P\_T0\_i} = P\_T0\_i - P\_T1\_i" /></a>
   
   此时我们将对应的样本i的输出值更改为上面的梯度，让下一个模型T2去拟合这个数值。然后依次迭代。最后我们所有树的预测值加起来就ok了。
  
       举个例子，如果样本1的输出真实值为10，树T1针对样本1的预测值为18，
       然后我们让树T2去拟合样本1的值为10-18=-8。如果树T2的输出值为-10，
       我们再让树T3去拟合-8-(-10)=2，结果树T3的预测值为1。
       如果到此迭代结束，在最终对样本1的预测值为：18+(-10)+1=9。
  
   到这里，整个GBDT的流程就大致清楚了。GBDT通过多轮迭代,每轮迭代产生一个弱模型，每个模型都是在上一个模型的残差基础上进行训练。通过更改计算树的输出值的方式，以及损失函数的定义得出不同的梯度形式，从而得到多种形式的GBDT。因为梯度是根据损失函数计算而来的，每一个新建立的树都是在沿着梯度的方向减小损失函数，这就是GBDT的核心。
   
   
    * **GBDT流程图** 
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/gbdt.png)
   
   
 * **GBDT步骤** 
 
    * **回归问题**
   
       + **训练数据集**
    
    
    
   
