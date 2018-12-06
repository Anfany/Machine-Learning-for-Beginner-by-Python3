# XGBoost  

* **XGBoost程序文件**

* **XGBoost目标函数**

    首先认清一点，它是GBDT的升级版，在效率、方法方面都进行了优化。
    
    不管对于回归问题还是分类问题，好的机器学习方法的目的就是降低目标函数(也可称为损失函数)的值，目标函数包括2个部分：一是模型的损失函数，二是模型的复杂度。也就是目标函数具有下面的形式：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{C}=\mathbf{F}(Y,\tilde{Y})&plus;\mathbf{G}(\mathbf{model})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{C}=\mathbf{F}(Y,\tilde{Y})&plus;\mathbf{G}(\mathbf{model})" title="\mathbf{C}=\mathbf{F}(Y,\tilde{Y})+\mathbf{G}(\mathbf{model})" /></a>
    
    上面公式中，前者表示模型的损失函数的值，降低它是为了降低偏差，也就是使得预测的数据和真实的数据更为接近；后者表示这个模型的复杂度，是为了降低方差，增强模型的泛化能力。
    
    对于XGBoost框架而言，以弱模型选择决策树(因为在XGBoost框架中，弱模型可以有多种选择，而在GBDT中，弱模型就是决策树)为例，来说明XGBoost，其目标函数为：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;where&space;\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})&plus;&space;\sum_{j=1}^{k}\mathbf{G}(T\_j)\\&space;\\&space;where&space;\;&space;\;&space;\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" title="\\ \mathbf{C}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,\tilde{Y\_i})+ \sum_{j=1}^{k}\mathbf{G}(T\_j)\\ \\ where \; \; \mathbf{G}(T\_j) = \gamma * \mathbf{H}(T\_j) + \frac{1}{2}\lambda *\left \| w \right \|^{2}" /></a>
    
    其中**H**(T_j)表示树T_j的叶子节点的个数，w是叶子节点的输出数值。
    
 * **XGBoost目标函数求解**   
 
     现在说一下XGBoost是如何求解上述目标函数的最小值的。可以看出，上面的目标函数的参数其实也是一个函数(其实树可看作一个函数)，因此要求解需要换个思路。也就是Boosing，利用弱模型的加法形式转换上面的目标函数。此外，在这里利用贪心算法，也就是当前一代的损失函数比上一代的低就可以，不是从整体考虑。转换后的第t代的目标函数为：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C\_t}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1}&plus;\mathbf{f\_t}(X\_i))&plus;\mathbf{G}(\mathbf{f\_t})&plus;&space;G_{num}\\&space;\\&space;where&space;\:&space;G_{num}=\sum_{j=1}^{t-1}\mathbf{G(f\_j)}&space;\,&space;is&space;\:&space;a&space;\:&space;constant." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C\_t}=\sum_{i=1}^{n}\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1}&plus;\mathbf{f\_t}(X\_i))&plus;\mathbf{G}(\mathbf{f\_t})&plus;&space;G_{num}\\&space;\\&space;where&space;\:&space;G_{num}=\sum_{j=1}^{t-1}\mathbf{G(f\_j)}&space;\,&space;is&space;\:&space;a&space;\:&space;constant." title="\\ \mathbf{C\_t}=\sum_{i=1}^{n}\mathbf{F}(Y\_i, \tilde{Y}\_i_{t-1}+\mathbf{f\_t}(X\_i))+\mathbf{G}(\mathbf{f\_t})+ G_{num}\\ \\ where \: G_{num}=\sum_{j=1}^{t-1}\mathbf{G(f\_j)} \, is \: a \: constant." /></a>
     
     根据二阶泰勒展开公式，可得到上述目标函数的近似值：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{C\_t}&space;\simeq&space;\sum_{i=1}^{n}[\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1})&plus;\mathbf{g_{i}}*\mathbf{f\_t}(X\_i)&plus;\frac{1}{2}\mathbf{h_{i}}*\mathbf{f\_t}^{2}(X\_i)]&plus;\mathbf{G(f\_t)}&plus;G_{num}\\&space;\\&space;where&space;\:&space;\:&space;\mathbf{g_{i}}=\frac{\partial&space;\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1})}{\partial&space;\tilde{Y}\_i_{t-1}}&space;\:&space;and\:&space;\mathbf{h_{i}}=\frac{\partial^2&space;\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1})}{\partial&space;\tilde{Y}\_i_{t-1}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{C\_t}&space;\simeq&space;\sum_{i=1}^{n}[\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1})&plus;\mathbf{g_{i}}*\mathbf{f\_t}(X\_i)&plus;\frac{1}{2}\mathbf{h_{i}}*\mathbf{f\_t}^{2}(X\_i)]&plus;\mathbf{G(f\_t)}&plus;G_{num}\\&space;\\&space;where&space;\:&space;\:&space;\mathbf{g_{i}}=\frac{\partial&space;\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1})}{\partial&space;\tilde{Y}\_i_{t-1}}&space;\:&space;and\:&space;\mathbf{h_{i}}=\frac{\partial^2&space;\mathbf{F}(Y\_i,&space;\tilde{Y}\_i_{t-1})}{\partial&space;\tilde{Y}\_i_{t-1}}" title="\\ \mathbf{C\_t} \simeq \sum_{i=1}^{n}[\mathbf{F}(Y\_i, \tilde{Y}\_i_{t-1})+\mathbf{g_{i}}*\mathbf{f\_t}(X\_i)+\frac{1}{2}\mathbf{h_{i}}*\mathbf{f\_t}^{2}(X\_i)]+\mathbf{G(f\_t)}+G_{num}\\ \\ where \: \: \mathbf{g_{i}}=\frac{\partial \mathbf{F}(Y\_i, \tilde{Y}\_i_{t-1})}{\partial \tilde{Y}\_i_{t-1}} \: and\: \mathbf{h_{i}}=\frac{\partial^2 \mathbf{F}(Y\_i, \tilde{Y}\_i_{t-1})}{\partial \tilde{Y}\_i_{t-1}}" /></a>
     
     从上面的式子可以看出XGBoost和GBDT的差异，**GBDT只是利用了一阶导数，而XGBoost利用了一阶导数和二阶导数**。还有就是GBDT并没有涉及到模型的复杂度。这个式子求和中的第一项是个常数项，公式的最后一项也是常数项，将它们去掉，变为下式：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{O\_t}&space;=&space;\sum_{i=1}^{n}[\mathbf{g_{i}}*\mathbf{f\_t}(X\_i)&plus;\frac{1}{2}\mathbf{h_{i}}*\mathbf{f\_t}^{2}(X\_i)]&plus;\mathbf{G(f\_t)}\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{O\_t}&space;=&space;\sum_{i=1}^{n}[\mathbf{g_{i}}*\mathbf{f\_t}(X\_i)&plus;\frac{1}{2}\mathbf{h_{i}}*\mathbf{f\_t}^{2}(X\_i)]&plus;\mathbf{G(f\_t)}\\" title="\\ \mathbf{O\_t} = \sum_{i=1}^{n}[\mathbf{g_{i}}*\mathbf{f\_t}(X\_i)+\frac{1}{2}\mathbf{h_{i}}*\mathbf{f\_t}^{2}(X\_i)]+\mathbf{G(f\_t)}\\" /></a>
     
     从上面式子可知，目标函数只是与损失函数的一阶、二阶导数有关系，因此**XGBoost支持自定义的损失函数**。
 
    下面说明树的复杂度部分，回顾下复杂度的函数：
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{G}(T\_j)&space;=&space;\gamma&space;*&space;\mathbf{H}(T\_j)&space;&plus;&space;\frac{1}{2}\lambda&space;*\left&space;\|&space;w&space;\right&space;\|^{2}" title="\mathbf{G}(T\_j) = \gamma * \mathbf{H}(T\_j) + \frac{1}{2}\lambda *\left \| w \right \|^{2}" /></a>
    
    其中**H**(T_j)表示树T_j的叶子节点的个数，w是叶子节点的输出数值。
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    
    
    
    
    
