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
         
         Y0为输出值序列，X0为输入的向量集合。
         
       + **定义损失函数**
       
          1. **MSE(GBDT回归所用)**
          
             <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\frac{1}{2}&space;\sum_{i=1}^{n}(Y\_real\_i-Y\_tree\_i)^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\frac{1}{2}&space;\sum_{i=1}^{n}(Y\_real\_i-Y\_tree\_i)^{2}" title="\mathbf{Cost(Y\_real,Y\_tree\)} =\frac{1}{2} \sum_{i=1}^{n}(Y\_real\_i-Y\_tree\_i)^{2}" /></a>
          
          2. **绝对损失**
          
             <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\sum_{i=1}^{n}|Y\_real\_i-Y\_tree\_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\sum_{i=1}^{n}|Y\_real\_i-Y\_tree\_i|" title="\mathbf{Cost(Y\_real,Y\_tree\)} =\sum_{i=1}^{n}|Y\_real\_i-Y\_tree\_i|" /></a>
          
           3. **Huber损失**
           
              它是MSE和绝对损失的组合形式，对于远离中心的异常点，采用绝对损失，其他的点采用MSE。
              
              这个界限一般用分位数点度量。公式如下
            
               <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\left\{\begin{matrix}&space;\frac{1}{2}\sum_{i=1}^{n}(Y\_real\_i&space;-&space;Y\_tree\_i)^{2},|Y\_real\_i&space;-&space;Y\_tree\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;(|Y\_real\_i&space;-&space;Y\_tree\_i|-\frac{\delta&space;}{2}),|Y\_real\_i&space;-&space;Y\_tree\_i|>&space;\delta&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{Cost(Y\_real,Y\_tree\)}&space;=\left\{\begin{matrix}&space;\frac{1}{2}\sum_{i=1}^{n}(Y\_real\_i&space;-&space;Y\_tree\_i)^{2},|Y\_real\_i&space;-&space;Y\_tree\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;(|Y\_real\_i&space;-&space;Y\_tree\_i|-\frac{\delta&space;}{2}),|Y\_real\_i&space;-&space;Y\_tree\_i|>&space;\delta&space;\end{matrix}\right." title="\mathbf{Cost(Y\_real,Y\_tree\)} =\left\{\begin{matrix} \frac{1}{2}\sum_{i=1}^{n}(Y\_real\_i - Y\_tree\_i)^{2},|Y\_real\_i - Y\_tree\_i|\leq \delta \\ \\ \delta (|Y\_real\_i - Y\_tree\_i|-\frac{\delta }{2}),|Y\_real\_i - Y\_tree\_i|> \delta \end{matrix}\right." /></a>
            
       + **初始化：针对数据集Data0，建立第一个CART回归树T0**
       
           CART回归树T0，说是树，其实就只包括一个叶子节点。
           
           这个树的输出值就是使得损失函数值最小的数r，该树对于训练数据集的序列定义为Y_T0，其实这个序列的元素全是r：
           
           <a href="https://www.codecogs.com/eqnedit.php?latex=\\Cost(Y0,&space;\gamma)&space;\\s.t.\mathbf{min}(Y0)\leq&space;\gamma&space;\leq&space;\mathbf{max}(Y0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\Cost(Y0,&space;\gamma)&space;\\s.t.\mathbf{min}(Y0)\leq&space;\gamma&space;\leq&space;\mathbf{max}(Y0)" title="\\Cost(Y0, \gamma) \\s.t.\mathbf{min}(Y0)\leq \gamma \leq \mathbf{max}(Y0)" /></a>
           
           **当前模型为M0(x) = Y_T0**
       
       + **计算损失函数梯度**
       
           下面分别给出三种损失函数的负梯度在当前模型Y_T0下的值：
           
           
           1. **MSE**
           
           
                 <a href="https://www.codecogs.com/eqnedit.php?latex=G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y0,&space;Y\_T0)}}{\partial&space;Y\_T0\_i}&space;=&space;Y0\_i&space;-&space;Y\_T0\_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y0,&space;Y\_T0)}}{\partial&space;Y\_T0\_i}&space;=&space;Y0\_i&space;-&space;Y\_T0\_i" title="G\_i = - \frac{\partial \mathbf{Cost(Y0, Y\_T0)}}{\partial Y\_T0\_i} = Y0\_i - Y\_T0\_i" /></a>
              
           2. **绝对损失**
           
           
                <a href="https://www.codecogs.com/eqnedit.php?latex=G\_i&space;=&space;-\frac{\partial&space;\mathbf{Cost(Y0,&space;Y\_T0)}}{\partial&space;Y\_T0\_i}&space;=&space;\mathbf{sign}(Y0\_i&space;-&space;Y\_T0\_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G\_i&space;=&space;-\frac{\partial&space;\mathbf{Cost(Y0,&space;Y\_T0)}}{\partial&space;Y\_T0\_i}&space;=&space;\mathbf{sign}(Y0\_i&space;-&space;Y\_T0\_i)" title="G\_i = -\frac{\partial \mathbf{Cost(Y0, Y\_T0)}}{\partial Y\_T0\_i} = \mathbf{sign}(Y0\_i - Y\_T0\_i)" /></a>
          
           3. **Huber损失**
           
              
                <a href="https://www.codecogs.com/eqnedit.php?latex=G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y0,&space;Y\_T0)}}{\partial&space;Y\_T0\_i}&space;=&space;\left\{\begin{matrix}&space;Y0\_i&space;-&space;Y\_T0\_i,|Y0\_i&space;-&space;Y\_T0\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;*\mathbf{sign}(Y0\_i&space;-&space;Y\_T0\_i),|Y0\_i&space;-&space;Y\_T0\_i|>&space;\delta&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?G\_i&space;=&space;-&space;\frac{\partial&space;\mathbf{Cost(Y0,&space;Y\_T0)}}{\partial&space;Y\_T0\_i}&space;=&space;\left\{\begin{matrix}&space;Y0\_i&space;-&space;Y\_T0\_i,|Y0\_i&space;-&space;Y\_T0\_i|\leq&space;\delta&space;\\&space;\\&space;\delta&space;*\mathbf{sign}(Y0\_i&space;-&space;Y\_T0\_i),|Y0\_i&space;-&space;Y\_T0\_i|>&space;\delta&space;\end{matrix}\right." title="G\_i = - \frac{\partial \mathbf{Cost(Y0, Y\_T0)}}{\partial Y\_T0\_i} = \left\{\begin{matrix} Y0\_i - Y\_T0\_i,|Y0\_i - Y\_T0\_i|\leq \delta \\ \\ \delta *\mathbf{sign}(Y0\_i - Y\_T0\_i),|Y0\_i - Y\_T0\_i|> \delta \end{matrix}\right." /></a>
              
      + **生成新的数据集Data1** 
      
         <a href="https://www.codecogs.com/eqnedit.php?latex=Data1=\{(Xi,Yi)|Xi=X0[i],&space;Yi=G\_i\&space;,i=1,2,\cdots,n&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data1=\{(Xi,Yi)|Xi=X0[i],&space;Yi=G\_i\&space;,i=1,2,\cdots,n&space;\}" title="Data1=\{(Xi,Yi)|Xi=X0[i], Yi=G\_i\ ,i=1,2,\cdots,n \}" /></a>
         
      + **建立CART回归树T1** 
      
        针对新生成的数据集Data1，在建立CART回归树T1。
        
        回归树T1建立完成后，每个叶子节点的输出值不再设定为平均值，
        
        而是利用和初始化时相同的方法，对每一个叶子节点，
        
        计算使得下式具有最小值的值作为该叶子节点的输出值：
        
        下面以叶子节点h为例。
        
        <a href="https://www.codecogs.com/eqnedit.php?latex=\\Cost(Y0\_T1\_h,&space;\gamma&space;&plus;&space;c\_T1\_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\Cost(Y0\_T1\_h,&space;\gamma&space;&plus;&space;c\_T1\_h)" title="\\Cost(Y0\_T1\_h, \gamma + c\_T1\_h)" /></a>
        
        其中Y0_T1_h 表示树T1的叶子节点h中包含的样本的真实输出值，

        c_T1_h为树T1的叶子节点h的最佳的输出值。训练数据集总的输出定义为c_T1
        
        上述函数中前面的项是样本的真实输出值，这一点是不变的。
        
        后面的项就是目前所有模型的集成值。
        
        **当前模型为M1(x) = Y_T0 + c_T1**
        
      + **迭代** 
        
        计算损失函数的负梯度在当前模型M1(x)下的值，然后更新训练数据集的输出值，
        
        然后再建立回归树，再计算每个叶子节点的最佳输出，
        
        然后得到当前模型M2(X)，以此迭代，直到满足条件结束。
        
        
      + **结果集成** 
      
        假设对应m个弱模型的预测的结果分别为Pre_1, Pre_2,……,Pre_m,
        将所有回归树的结果相加，
        
        即为最终的结果，如下公式。

        <a href="https://www.codecogs.com/eqnedit.php?latex=Pre(m)&space;=&space;\sum_{j=1}^{m}Pre\_j\\&space;Pre(k)&space;=&space;Pre(k-1)&plus;Pre\_k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pre(m)&space;=&space;\sum_{j=1}^{m}Pre\_j\\&space;Pre(k)&space;=&space;Pre(k-1)&plus;Pre\_k" title="Pre(m) = \sum_{j=1}^{m}Pre\_j\\ Pre(k) = Pre(k-1)+Pre\_k" /></a>
        
        为了防止过拟合，可添加正则化的环节，

        <a href="https://www.codecogs.com/eqnedit.php?latex=Pre(k)&space;=&space;Pre(k-1)&plus;\alpha&space;*Pre\_k,&space;k\geq&space;2,0<\alpha&space;\leq&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Pre(k)&space;=&space;Pre(k-1)&plus;\alpha&space;*Pre\_k,&space;k\geq&space;2,0<\alpha&space;\leq&space;1" title="Pre(k) = Pre(k-1)+\alpha *Pre\_k, k\geq 2,0<\alpha \leq 1" /></a>，a为正则化因子。
        
        
    * **分类问题**
    
       这里解释下，为什么GBDT解决分类问题用的也是回归树。因为GBDT的根本就是在不断地拟合残差，残差的加减可以逐渐减小偏差。而分类树输出的是类别，他们之间的加减是没有意义的。分类问题和回归问题的最大不同在于损失函数的定义以及节点输出值的确定，其他都是一样的。
       
       + **二分类**
       
         + **对数似然损失函数**
         
           <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{Cost(Y\_r,&space;F(X))}&space;=&space;\mathbf{log}(1&plus;\mathbf{exp}(-2*\mathbf{Y\_r}*\mathbf{F(X)})),&space;\mathbf{Y\_r}=1&space;\mathbf{or}-1\\&space;\\&space;\mathbf{F(X)}&space;=&space;\frac{1}{2}\mathbf{log}(\frac{P(Y=1|X)}{P(Y=-1|X)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{Cost(Y\_r,&space;F(X))}&space;=&space;\mathbf{log}(1&plus;\mathbf{exp}(-2*\mathbf{Y\_r}*\mathbf{F(X)})),&space;\mathbf{Y\_r}=1&space;\mathbf{or}-1\\&space;\\&space;\mathbf{F(X)}&space;=&space;\frac{1}{2}\mathbf{log}(\frac{P(Y=1|X)}{P(Y=-1|X)})" title="\\ \mathbf{Cost(Y\_r, F(X))} = \mathbf{log}(1+\mathbf{exp}(-2*\mathbf{Y\_r}*\mathbf{F(X)})), \mathbf{Y\_r}=1 \mathbf{or}-1\\ \\ \mathbf{F(X)} = \frac{1}{2}\mathbf{log}(\frac{P(Y=1|X)}{P(Y=-1|X)})" /></a>
           
         + **计算负梯度**
         
            <a href="https://www.codecogs.com/eqnedit.php?latex=\\G\_i&space;=&space;-\frac{\partial&space;\mathbf{Cost(Y\_r,&space;F(X))}}{\partial&space;F(X\_i)}&space;=&space;\frac{\mathbf{2Y\_r\_i}}{1&space;&plus;&space;\mathbf{exp}(\mathbf{2Y\_r\_i}*F(X\_i))}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\G\_i&space;=&space;-\frac{\partial&space;\mathbf{Cost(Y\_r,&space;F(X))}}{\partial&space;F(X\_i)}&space;=&space;\frac{\mathbf{2Y\_r\_i}}{1&space;&plus;&space;\mathbf{exp}(\mathbf{2Y\_r\_i}*F(X\_i))}" title="\\G\_i = -\frac{\partial \mathbf{Cost(Y\_r, F(X))}}{\partial F(X\_i)} = \frac{\mathbf{2Y\_r\_i}}{1 + \mathbf{exp}(\mathbf{2Y\_r\_i}*F(X\_i))}" /></a>
         
         + **计算树的叶子节点的最佳的输出值**
         
              对于树Ta的第h个叶子节点，其输出值为：
             
              <a href="https://www.codecogs.com/eqnedit.php?latex=C\_Ta\_h&space;=&space;\frac{&space;\sum_{i\subseteq&space;Ta\_h}&space;G\_i\_Ta\_h}{&space;\sum_{i\subseteq&space;Ta\_h}[|&space;G\_i\_Ta\_h|*(2-|G\_i\_Ta\_h|)]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C\_Ta\_h&space;=&space;\frac{&space;\sum_{i\subseteq&space;Ta\_h}&space;G\_i\_Ta\_h}{&space;\sum_{i\subseteq&space;Ta\_h}[|&space;G\_i\_Ta\_h|*(2-|G\_i\_Ta\_h|)]}" title="C\_Ta\_h = \frac{ \sum_{i\subseteq Ta\_h} G\_i\_Ta\_h}{ \sum_{i\subseteq Ta\_h}[| G\_i\_Ta\_h|*(2-|G\_i\_Ta\_h|)]}" /></a>
             
             
         +  **集成**
         
             迭代完成后，得到的模型为FM，对于样本X，分别计算其属于1和-1的概率：
             
             <a href="https://www.codecogs.com/eqnedit.php?latex=\\P(Y=1|X)=&space;\frac{1}{1&plus;exp(-2F\_m(X))}\\&space;\\&space;P(Y=-1|X)=&space;\frac{1}{1&plus;exp(2F\_m(X))}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\P(Y=1|X)=&space;\frac{1}{1&plus;exp(-2F\_m(X))}\\&space;\\&space;P(Y=-1|X)=&space;\frac{1}{1&plus;exp(2F\_m(X))}" title="\\P(Y=1|X)= \frac{1}{1+exp(-2F\_m(X))}\\ \\ P(Y=-1|X)= \frac{1}{1+exp(2F\_m(X))}" /></a>
             
             概率较大的对应的类别就是预测类别。

          
       + **多分类**
         
    
    
    
   
