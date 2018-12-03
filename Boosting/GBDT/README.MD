# GBDT


* **GBDT程序文件**

* **GBDT说明**

   **GBDT**是Gradient Boosting Decison Tree的简称，其中Gradient是梯度，是这个方法的核心；Boosting是提升，是这个方法的框架；Decision Tree是决策树，是实现这个方法用到的模型。

   GBDT可以解决回归问题，经过一些处理也可以解决分类(二类、多类)问题，但是用到的树都是回归树，这一点需要牢记。
  
   首先通过简单的回归例子说明一下**提升(Boosting)树**：

       举个例子，如果样本1的输出真实值为10，树T1针对样本1的预测值为18，
       然后我们让树T2去拟合样本1的值为10-18=-8(残差)。如果树T2的输出值为-10，
       我们再让树T3去拟合-8-(-10)=2(残差)，结果树T3的预测值为1。
       如果到此迭代结束，在最终对样本1的预测值为：18+(-10)+1=9。
  
   到这里，提升回归树的流程就大致清楚了。也就是通过多轮迭代，每轮迭代产生一个弱模型，每个模型都是在上一个模型的残差基础上进行训练的，最后将所有树的结果求和得出最终的结果。
   
   GBDT就是在提升树的基础上，利用了梯度提升的方法，也就是用损失函数的**负梯度**在当前模型下的值来作为提升树中残差的近似值。对于GBDT用于回归问题而言，如果损失函数定义为MSE，则其负梯度就是残差。因此残差是损失函数负梯度的一种特殊情况。负梯度是残差这种思想的一般化。残差只可以用于回归问题，但是这种负梯度的思想也可用于分类问题。
   
    * **GBDT流程图** 
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/gbdt.png)
   
   
 * **GBDT步骤** 
 
    * **回归问题**
   
       + **训练数据集**
       
         <a href="https://www.codecogs.com/eqnedit.php?latex=Data0&space;=&space;\{(X1,Y1),&space;(X2,Y2),\cdots&space;,(Xn,Yn)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data0&space;=&space;\{(X1,Y1),&space;(X2,Y2),\cdots&space;,(Xn,Yn)\}" title="Data0 = \{(X1,Y1), (X2,Y2),\cdots ,(Xn,Yn)\}" /></a>
         
         Y_T0为输出值序列，X0为输入的向量集合。
         
       + **定义损失函数**
       
          1. **MSE(GBDT回归所用)**
          
             <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\frac{1}{2}&space;\sum_{i=1}^{n}(Y\_real\_i-Y\_tree\_i)^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\frac{1}{2}&space;\sum_{i=1}^{n}(Y\_real\_i-Y\_tree\_i)^{2}" title="\mathbf{Cost(Y\_real,Y\_tree\)} =\frac{1}{2} \sum_{i=1}^{n}(Y\_real\_i-Y\_tree\_i)^{2}" /></a>
          
          2. **绝对损失**
          
             <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\sum_{i=1}^{n}|Y\_real\_i-Y\_tree\_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\sum_{i=1}^{n}|Y\_real\_i-Y\_tree\_i|" title="\mathbf{Cost(Y\_real,Y\_tree\)} =\sum_{i=1}^{n}|Y\_real\_i-Y\_tree\_i|" /></a>
          
           3. **Huber损失**
           
              它是MSE和绝对损失的组合形式，对于远离中心的异常点，采用绝对损失，其他的点采用MSE。
              
              这个界限一般用分位数点度量。公式如下
            
               <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\left\{\begin{matrix}&space;\frac{1}{2}\sum_{i=1}^{n}(Y\_real\_i&space;-&space;Y\_tree\_i)^{2},|Y\_real\_i&space;-&space;Y\_tree\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;(|Y\_real\_i&space;-&space;Y\_tree\_i|-\frac{\delta&space;}{2}),|Y\_real\_i&space;-&space;Y\_tree\_i|>&space;\delta&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\left\{\begin{matrix}&space;\frac{1}{2}\sum_{i=1}^{n}(Y\_real\_i&space;-&space;Y\_tree\_i)^{2},|Y\_real\_i&space;-&space;Y\_tree\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;(|Y\_real\_i&space;-&space;Y\_tree\_i|-\frac{\delta&space;}{2}),|Y\_real\_i&space;-&space;Y\_tree\_i|>&space;\delta&space;\end{matrix}\right." title="\mathbf{Cost(Y\_real,Y\_tree\)} =\left\{\begin{matrix} \frac{1}{2}\sum_{i=1}^{n}(Y\_real\_i - Y\_tree\_i)^{2},|Y\_real\_i - Y\_tree\_i|\leq \delta \\ \\ \delta (|Y\_real\_i - Y\_tree\_i|-\frac{\delta }{2}),|Y\_real\_i - Y\_tree\_i|> \delta \end{matrix}\right." /></a>
            
       + **针对数据集Data0，建立第一个CART回归树T1**
       
           T1模型下，训练数据集对应的输出为Y_T1。
       
       + **计算损失函数梯度**
       
           下面分别给出三种损失函数的负梯度在当前模型Y_T1下的值：
           
           
           1. **MSE**
           
           
                 <a href="https://www.codecogs.com/eqnedit.php?latex=G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y\_T0,&space;Y\_T1)}}{\partial&space;Y\_T1\_i}&space;=&space;Y\_T0\_i&space;-&space;Y\_T1\_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y\_T0,&space;Y\_T1)}}{\partial&space;Y\_T1\_i}&space;=&space;Y\_T0\_i&space;-&space;Y\_T1\_i" title="G\_i = - \frac{\partial \mathbf{Cost(Y\_T0, Y\_T1)}}{\partial Y\_T1\_i} = Y\_T0\_i - Y\_T1\_i" /></a>
              
           2. **绝对损失**
           
           
                <a href="https://www.codecogs.com/eqnedit.php?latex=G\_i&space;=&space;-\frac{\partial&space;\mathbf{Cost(Y\_T0,&space;Y\_T1)}}{\partial&space;Y\_T1\_i}&space;=&space;\mathbf{sign}(Y\_T0\_i&space;-&space;Y\_T1\_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G\_i&space;=&space;-\frac{\partial&space;\mathbf{Cost(Y\_T0,&space;Y\_T1)}}{\partial&space;Y\_T1\_i}&space;=&space;\mathbf{sign}(Y\_T0\_i&space;-&space;Y\_T1\_i)" title="G\_i = -\frac{\partial \mathbf{Cost(Y\_T0, Y\_T1)}}{\partial Y\_T1\_i} = \mathbf{sign}(Y\_T0\_i - Y\_T1\_i)" /></a>
          
           3. **Huber损失**
           
              
                <a href="https://www.codecogs.com/eqnedit.php?latex=G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y\_T0,&space;Y\_T1)}}{\partial&space;Y\_T1\_i}&space;=&space;\left\{\begin{matrix}&space;Y\_T0\_i&space;-&space;Y\_T1\_i,|Y\_T0\_i&space;-&space;Y\_T1\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;*\mathbf{sign}(Y\_T0\_i&space;-&space;Y\_T1\_i),|Y\_T0\_i&space;-&space;Y\_T1\_i|>&space;\delta&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y\_T0,&space;Y\_T1)}}{\partial&space;Y\_T1\_i}&space;=&space;\left\{\begin{matrix}&space;Y\_T0\_i&space;-&space;Y\_T1\_i,|Y\_T0\_i&space;-&space;Y\_T1\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;*\mathbf{sign}(Y\_T0\_i&space;-&space;Y\_T1\_i),|Y\_T0\_i&space;-&space;Y\_T1\_i|>&space;\delta&space;\end{matrix}\right." title="G\_i = - \frac{\partial \mathbf{Cost(Y\_T0, Y\_T1)}}{\partial Y\_T1\_i} = \left\{\begin{matrix} Y\_T0\_i - Y\_T1\_i,|Y\_T0\_i - Y\_T1\_i|\leq \delta \\ \\ \delta *\mathbf{sign}(Y\_T0\_i - Y\_T1\_i),|Y\_T0\_i - Y\_T1\_i|> \delta \end{matrix}\right." /></a>
              
      + **生成新的数据集Data1** 
      
         <a href="https://www.codecogs.com/eqnedit.php?latex=Data1=\{(Xi,Yi)|Xi=X0[i],&space;Yi=G\_i\&space;,i=1,2,\cdots,n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data1=\{(Xi,Yi)|Xi=X0[i],&space;Yi=G\_i\&space;,i=1,2,\cdots,n&space;\}" title="Data1=\{(Xi,Yi)|Xi=X0[i], Yi=G\_i\ ,i=1,2,\cdots,n \}" /></a>
         
      + **迭代** 
      
        针对新生成的数据集Data1，在建立CART回归树T2。如此迭代直到满足结束条件。
        
      + **结果集成** 
      
        假设对应m个弱模型的预测的结果分别为Pre_T1, Pre_T2,……,Pre_Tm,
        将所有回归树的结果相加，
        
        即为最终的结果，如下公式。

        <a href="https://www.codecogs.com/eqnedit.php?latex=Pre(m)&space;=&space;\sum_{j=1}^{m}Pre\_j\\&space;Pre(k)&space;=&space;Pre(k-1)&plus;Pre\_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pre(m)&space;=&space;\sum_{j=1}^{m}Pre\_j\\&space;Pre(k)&space;=&space;Pre(k-1)&plus;Pre\_k" title="Pre(m) = \sum_{j=1}^{m}Pre\_j\\ Pre(k) = Pre(k-1)+Pre\_k" /></a>
        
        为了防止过拟合，可添加正则化的环节，

        <a href="https://www.codecogs.com/eqnedit.php?latex=Pre(k)&space;=&space;Pre(k-1)&plus;\alpha&space;*Pre\_k,&space;k\geq&space;2,0<\alpha&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pre(k)&space;=&space;Pre(k-1)&plus;\alpha&space;*Pre\_k,&space;k\geq&space;2,0<\alpha&space;\leq&space;1" title="Pre(k) = Pre(k-1)+\alpha *Pre\_k, k\geq 2,0<\alpha \leq 1" /></a>，a为正则化因子。
        
        
    * **回归问题**
    
       + **二分类**
       
       
          
       + **多分类**
         
    
    
    
   
