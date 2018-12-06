# XGBoost  

* **XGBoost程序文件**

* **XGBoost目标函数**

    首先认清一点，它是GBDT的升级版，在效率、方法方面都进行了优化。
    
    不管对于回归问题还是分类问题，好的机器学习方法的目的就是降低目标函数(也可称为损失函数)的值，目标函数包括2个部分：一是模型的损失函数，二是模型的复杂度。也就是目标函数具有下面的形式：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{C}=\mathbf{F}(Y,\tilde{Y})&plus;\mathbf{G}(\mathbf{model})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{C}=\mathbf{F}(Y,\tilde{Y})&plus;\mathbf{G}(\mathbf{model})" title="\mathbf{C}=\mathbf{F}(Y,\tilde{Y})+\mathbf{G}(\mathbf{model})" /></a>
    
    上面公式中，前者表示模型的损失函数的值，降低它是为了降低偏差，也就是使得预测的数据和真实的数据更为接近；后者表示这个模型的复杂度，是为了降低方差，增强模型的泛化能力。
    
    对于XGBoost框架而言，以弱模型选择决策树(因为在XGBoost框架中，弱模型可以有多种选择，而在GBDT中，弱模型就是决策树)为例，来说明XGBoost，其目标函数为：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" title="\\ \mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})+ \sum_{j=1}^{k}\mathbf{G}(T\_j)\\ \\ s.t.\; \; \mathbf{G}(T\_j) = \gamma * \mathbf{H}(T\_j) + \frac{1}{2}\lambda *\left \| w \right \|^{2}" /></a>
    
    其中**H**(T_j)表示树T_j的叶子节点的个数，w是叶子节点的输出数值。
    
 * **XGBoost目标函数求解**   
 
     现在说以下XGBoost是如何求解上述目标函数的最小值的。下面以回归问题为例，其实回归问题和分类问题的目标函数的不同就在于损失函数。下面给出回归问题的目标函数：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C}=\sum_{i=1}^{n}(Y\_i-\tilde{Y\_i})^{2}&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C}=\sum_{i=1}^{n}(Y\_i-\tilde{Y\_i})^{2}&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;s.t.\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" title="\\ \mathbf{C}=\sum_{i=1}^{n}(Y\_i-\tilde{Y\_i})^{2}+ \sum_{j=1}^{k}\mathbf{G}(T\_j)\\ \\ s.t.\; \; \mathbf{G}(T\_j) = \gamma * \mathbf{H}(T\_j) + \frac{1}{2}\lambda *\left \| w \right \|^{2}" /></a>
     
     不难看出，上面的目标函数的参数也是一个函数，因此要求解需要换个思路。也就是Boosing，利用弱模型的加法形式转换上面的目标函数：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C\_t}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1}&plus;\mathbf{f\_t}(X))&plus;\mathbf{G}(\mathbf{f\_t})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C\_t}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1}&plus;\mathbf{f\_t}(X))&plus;\mathbf{G}(\mathbf{f\_t})" title="\\ \mathbf{C\_t}=\sum_{i=1}^{n}\mathbf{F}(Y\_i, \tilde{Y}\_i_{t-1}+\mathbf{f\_t}(X))+\mathbf{G}(\mathbf{f\_t})" /></a>
     
     
     
     
    
    
    
    
    
