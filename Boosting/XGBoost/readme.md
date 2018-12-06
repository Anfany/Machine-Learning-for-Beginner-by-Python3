# XGBoost  

* **XGBoost程序文件**

* **XGBoost目标函数**

    首先认清一点，它是GBDT的升级版，在效率、方法方面都进行了优化。
    
    不管对于回归问题还是分类问题，好的机器学习方法的目的就是降低目标函数的值，目标函数包括2个部分：一是损失函数(偏差),二是模型的复杂度(方差)。也就是目标函数具有下面的形式：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{C}=\mathbf{F}(Y,\tilde{Y})&plus;\mathbf{M}(\mathbf{F})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{C}=\mathbf{F}(Y,\tilde{Y})&plus;\mathbf{M}(\mathbf{F})" title="\mathbf{C}=\mathbf{F}(Y,\tilde{Y})+\mathbf{M}(\mathbf{F})" /></a>
    
    上面公式中，前者表示损失函数的值，是为了降低偏差，也就是使得预测的数据和真实的数据更为接近；后者表示这个模型的复杂度，是为了降低方差，增强模型的泛化能力。对于XGBoost框架而言，以弱模型选择决策树(因为在这个框架中，弱模型可以有多种选择，而在GBDT中，弱模型就是决策树)为例，来说明XGBoost，其损失函数为：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" title="\\ \mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})+ \sum_{j=1}^{k}\mathbf{G}(T\_j)\\ \\ s.t.\; \; \mathbf{G}(T\_j) = \gamma * \mathbf{H}(T\_j) + \frac{1}{2}\lambda *\left \| w \right \|^{2}" /></a>
    
    其中**H**(T_j)表示树T_j的叶子节点的个数，w是只叶子节点的输出数值。
    
 * **XGBoost目标函数求解**   
 
     现在说以下XGBoost是如何求解上述目标函数的最小值的。下面以回归问题为例，其实回归问题和分类问题的目标函数的主要不同就是损失函数。下面给出回归问题的目标函数：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C}=\sum_{i=1}^{n}(Y\_i-\tilde{Y\_i})^{2}&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C}=\sum_{i=1}^{n}(Y\_i-\tilde{Y\_i})^{2}&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" title="\\ \mathbf{C}=\sum_{i=1}^{n}(Y\_i-\tilde{Y\_i})^{2}+ \sum_{j=1}^{k}\mathbf{G}(T\_j)\\ \\ s.t.\; \; \mathbf{G}(T\_j) = \gamma * \mathbf{H}(T\_j) + \frac{1}{2}\lambda *\left \| w \right \|^{2}" /></a>
     
     
     
     
     
     
    
    
    
    
    
