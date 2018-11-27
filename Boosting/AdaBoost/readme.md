## AdaBoost

* **AdaBoost初识**

这个方法主要涉及到2个权重集合：

1. **样本的权重集合Y**：每个样本都对应一个权重。 在构建第一个弱模型之前，所有的训练样本的权重是一样的。第一个模型完成后，要加大那些被这个模型错误分类(分类问题)、或者说预测真实差值较大(回归问题)的样本的权重。依次迭代，最终构建多个弱模型。每个弱模型所对应的训练数据集样本是一样的，只是数据集中的样本权重是不一样的。

2. **弱模型的权重集合M**：得到的每个弱模型都对应一个权重。这个权重依赖于每个弱模型的精度(分类：正确率，回归：MSE)。精度越高的模型，其权重也就越大，在最终集成结果时，其话语权也就越大。

* **AdaBoost步骤**

  * **分类问题**
  
   * **训练数据集**
    
      <a href="https://www.codecogs.com/eqnedit.php?latex=Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" title="Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" /></a>，假设Yi=1 or -1，这种定义便于后面的结果集成。
   
   * **初始的样本权重集合S0，弱模型的权重集合为D**
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=S0=\{S0i=\frac{1}{n},i=1,2,\cdots&space;n\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S0=\{S0i=\frac{1}{n},i=1,2,\cdots&space;n\}" title="S0=\{S0i=\frac{1}{n},i=1,2,\cdots n\}" /></a>
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=D&space;=&space;\{D1,&space;D2,&space;\cdots&space;,&space;Dm\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D&space;=&space;\{D1,&space;D2,&space;\cdots&space;,&space;Dm\}" title="D = \{D1, D2, \cdots , Dm\}" /></a>
       
      n为数据集样本个数，m为要建立的弱模型的个数
   
    * **针对数据集构建弱模型M1，得到这个弱模型的错误率为**
   
            <a href="https://www.codecogs.com/eqnedit.php?latex=e&space;=&space;\frac{C_{error}}{C_{Data}}&space;=&space;\frac{C_{error}}{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e&space;=&space;\frac{C_{error}}{C_{Data}}&space;=&space;\frac{C_{error}}{n}" title="e = \frac{C_{error}}{C_{Data}} = \frac{C_{error}}{n}" /></a>
       ，其中Cerror为被错分的样本个数，CData为全部的样本个数，也就是n。
   
     * **计算该模型的权重**
   
        <a href="https://www.codecogs.com/eqnedit.php?latex=D1=\frac{1}{2}\mathbf{ln}(\frac{e}{1-e})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D1=\frac{1}{2}\mathbf{ln}(\frac{e}{1-e})" title="D1=\frac{1}{2}\mathbf{ln}(\frac{e}{1-e})" /></a>
   
   * **更改被错分的样本的权重**
   
   
   
   
   
  
  
  
  * **回归问题**  
  
   
* **AdaBoost答疑**   
   1. 这个弱模型可以是什么？
   
   答：经常用的就是单层的决策树，也称为决策树桩(Decision Stump)，例如单层的CART。其实这个层数也是参数，需要交叉验证得到最好的。当然这个弱模型也可以是SVM、逻辑回归、神经网络等。
   
   2. 增加的样本权重如何在下一个模型训练时体现出作用？
   
   答：通过重复样本的方式，就是增加上一个模型被错分的样本的个数。
  
