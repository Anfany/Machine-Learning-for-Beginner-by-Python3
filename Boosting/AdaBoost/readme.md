## AdaBoost

* **AdaBoost初识**

这个方法主要涉及到2个权重集合：

1. **样本的权重集合**：每个样本都对应一个权重。 在构建第一个弱模型之前，所有的训练样本的权重是一样的。第一个模型完成后，要加大那些被这个模型错误分类(分类问题)、或者说预测真实差值较大(回归问题)的样本的权重。依次迭代，最终构建多个弱模型。每个弱模型所对应的训练数据集样本是一样的，只是数据集中的样本权重是不一样的。

2. **弱模型的权重集合**：得到的每个弱模型都对应一个权重。精度越高(分类问题的错分率越低，回归问题的错误率越低)的模型，其权重也就越大，在最终集成结果时，其话语权也就越大。

* **AdaBoost流程图**

![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/AdaBoost/model.png)

* **AdaBoost步骤**

  * **分类问题**
  
    * **训练数据集**
    
      <a href="https://www.codecogs.com/eqnedit.php?latex=Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" title="Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" /></a>，
      
      令Yi = 1 or -1，这种定义便于后面的结果集成。集合Y0表示数据集样本的真实类别序列。
      
   
    * **初始的样本权重集合S1，弱模型的权重集合为D**
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=S1=\{S1i=\frac{1}{n},i=1,2,\cdots&space;n\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S1=\{S1i=\frac{1}{n},i=1,2,\cdots&space;n\}" title="S1=\{S1i=\frac{1}{n},i=1,2,\cdots n\}" /></a>
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=D&space;=&space;\{D1,&space;D2,&space;\cdots&space;,&space;Dm\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D&space;=&space;\{D1,&space;D2,&space;\cdots&space;,&space;Dm\}" title="D = \{D1, D2, \cdots , Dm\}" /></a>,n为数据集样本个数，m为要建立的弱模型的个数
   
    * **针对数据集构建弱模型M1，得到这个弱模型的错分率为**
    
        假设弱模型M1的训练数据集的预测类别序列为P1，预测数据集的预测类别序列为Pre_1。
   
         <a href="https://www.codecogs.com/eqnedit.php?latex=err=\frac{C_{error}}{C_{Data}}=\frac{\left&space;\|&space;P1_{i}&space;\neq&space;Y0_{i},i=1,2\cdots&space;,n&space;\right&space;\|}{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?err=\frac{C_{error}}{C_{Data}}=\frac{\left&space;\|&space;P1_{i}&space;\neq&space;Y0_{i},i=1,2\cdots&space;,n&space;\right&space;\|}{n}" title="err=\frac{C_{error}}{C_{Data}}=\frac{\left \| P1_{i} \neq Y0_{i},i=1,2\cdots ,n \right \|}{n}" /></a>，
         
         其中Cerror表示被弱模型M1错分的样本个数，CData为全部的样本个数，也就是n。
   
     * **计算弱模型M1的权重**
   
        <a href="https://www.codecogs.com/eqnedit.php?latex=D1=\frac{1}{2}\mathbf{ln}(\frac{1-err}{err})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D1=\frac{1}{2}\mathbf{ln}(\frac{1-err}{err})" title="D1=\frac{1}{2}\mathbf{ln}(\frac{1-err}{err})" /></a>
   
    * **更改样本的权重**
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\\S2i=\frac{S1i*e^{-D1*Y0i*P1i}}{\mathbf{sum}(S)}=\left\{\begin{matrix}&space;\frac{S1i*e^{-D1}}{\mathbf{sum}(S)},&space;P1i=Y0i\\&space;\\&space;\frac{S1i*e^{D1}}{\mathbf{sum}(S)},&space;P1i\neq&space;Y0i\\&space;\end{matrix}\right.\\&space;\mathbf{sum}(S)=\sum_{i=1}^{n}&space;S1i*e^{-D1*Y0i*P1i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\\S2i=\frac{S1i*e^{-D1*Y0i*P1i}}{\mathbf{sum}(S)}=\left\{\begin{matrix}&space;\frac{S1i*e^{-D1}}{\mathbf{sum}(S)},&space;P1i=Y0i\\&space;\\&space;\frac{S1i*e^{D1}}{\mathbf{sum}(S)},&space;P1i\neq&space;Y0i\\&space;\end{matrix}\right.\\&space;\mathbf{sum}(S)=\sum_{i=1}^{n}&space;S1i*e^{-D1*Y0i*P1i}" title="\large \\S2i=\frac{S1i*e^{-D1*Y0i*P1i}}{\mathbf{sum}(S)}=\left\{\begin{matrix} \frac{S1i*e^{-D1}}{\mathbf{sum}(S)}, P1i=Y0i\\ \\ \frac{S1i*e^{D1}}{\mathbf{sum}(S)}, P1i\neq Y0i\\ \end{matrix}\right.\\ \mathbf{sum}(S)=\sum_{i=1}^{n} S1i*e^{-D1*Y0i*P1i}" /></a>。
       
       D1为非负数，因此预测正确的样本的权重会比上一次的降低，
       
       预测错误的会比上一次的增高。
    
       其中除以**sum**(S)，相当于将样本权重规范化。
       
    * **迭代**
    
       当达到设定的迭代次数时停止，或者错分率小于某个小的正数时停止迭代。
       
       此时得到m个弱模型，以及预测数据集对应的预测结果序列Pre_1，Pre_2， ……Pre_m，
       
       以及模型的权重集合D。
    
    * **结果集成**
    
     针对第i个预测样本的集成结果为JI_i,
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;JI\_i&space;=&space;\mathbf{sign}\sum_{k=1}^{m}Dk*Pre\_k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;JI\_i&space;=&space;\mathbf{sign}\sum_{k=1}^{m}Dk*Pre\_k_i" title="\large JI\_i = \mathbf{sign}\sum_{k=1}^{m}Dk*Pre\_k_i" /></a>，**sign**为符号函数。
    
  
  回归问题和分类问题的最大不同在于，回归问题错误率的计算不同于分类问题的错分率，下面给出回归问题的步骤，因为回归算法有很多的变种，这里以**Adaboost R2算法**为例说明：
  
  * **回归问题**  
  
    * **训练数据集**
    
      <a href="https://www.codecogs.com/eqnedit.php?latex=Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" title="Data=\{(X1,Y1),(X2,Y2),\cdots,(Xn,Yn)\}" /></a>，输出值的序列为Y0。
      
    * **初始的样本权重集合S1，弱模型的权重集合为D**
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=S1=\{S1i=\frac{1}{n},i=1,2,\cdots&space;n\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S1=\{S1i=\frac{1}{n},i=1,2,\cdots&space;n\}" title="S1=\{S1i=\frac{1}{n},i=1,2,\cdots n\}" /></a>
   
       <a href="https://www.codecogs.com/eqnedit.php?latex=D&space;=&space;\{D1,&space;D2,&space;\cdots&space;,&space;Dm\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D&space;=&space;\{D1,&space;D2,&space;\cdots&space;,&space;Dm\}" title="D = \{D1, D2, \cdots , Dm\}" /></a>,n为数据集样本个数，m为要建立的弱模型的个数
   
    * **针对数据集构建弱模型M1，得到这个弱模型的错误率为**
    
        假设弱模型M1的训练数据集的预测类别序列为P1，预测数据集的预测类别序列为Pre_1。
   
         <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;maxerr&space;=&space;\mathbf{max}&space;(|Y0i-P1i|,i=1,2,\cdots,n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;maxerr&space;=&space;\mathbf{max}&space;(|Y0i-P1i|,i=1,2,\cdots,n)" title="\large maxerr = \mathbf{max} (|Y0i-P1i|,i=1,2,\cdots,n)" /></a>
         
         * **误差损失为线性**
         
         <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;err_i=\frac{|Y0i-P1i|}{maxerr}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;err_i=\frac{|Y0i-P1i|}{maxerr}" title="\large err_i=\frac{|Y0i-P1i|}{maxerr}" /></a>
         
         * **误差损失为平方**
         
         <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;erri=\frac{(Y0i-P1i)^{2}}{maxerr^{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;erri=\frac{(Y0i-P1i)^{2}}{maxerr^{2}}" title="\large erri=\frac{(Y0i-P1i)^{2}}{maxerr^{2}}" /></a>
          
         * **误差损失为指数**
         
         <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;erri=1-e^{\frac{-|P1i-Y0i|}{maxerr}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;erri=1-e^{\frac{-|P1i-Y0i|}{maxerr}}" title="\large erri=1-e^{\frac{-|P1i-Y0i|}{maxerr}}" /></a>
         
        错误率的计算公式为：
        
      <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;err=\sum_{i=1}^{N}S1i*erri" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;err=\sum_{i=1}^{N}S1i*erri" title="\large err=\sum_{i=1}^{N}S1i*erri" /></a>
   
     * **计算弱模型M1的权重**
   
        <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;D1=\mathbf{ln}\frac{1-err}{err}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;D1=\mathbf{ln}\frac{1-err}{err}" title="\large D1=\mathbf{ln}\frac{1-err}{err}" /></a>
   
    * **更改样本的权重**
   
     <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\\S2i=\frac{S1i}{\mathbf{sum}(S)}*D1^{1-erri}\\&space;\mathbf{sum}(S)=\sum_{i=1}^{n}S1i*D1^{1-erri}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\\S2i=\frac{S1i}{\mathbf{sum}(S)}*D1^{1-erri}\\&space;\mathbf{sum}(S)=\sum_{i=1}^{n}S1i*D1^{1-erri}" title="\large \\S2i=\frac{S1i}{\mathbf{sum}(S)}*D1^{1-erri}\\ \mathbf{sum}(S)=\sum_{i=1}^{n}S1i*D1^{1-erri}" /></a> 。
    其中除以**sum**(S)，相当于将样本权重归一化。
       
    * **迭代**
    
     当达到设定的迭代次数时停止，或者错误率小于某个小的正数时停止迭代。
     
     此时得到m个弱模型，以及预测数据集对应的预测结果序列Pre_1，Pre_2， ……Pre_m，
     
     以及模型的权重集合D。
    
    * **结果集成**
    
       针对第i个预测样本的集成结果为JI_i,
    
       <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;JI\_i&space;=&space;\sum_{k=1}^{m}&space;Dk*Pre\_k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;JI\_i&space;=&space;\sum_{k=1}^{m}&space;Dk*Pre\_k_i" title="\large JI\_i = \sum_{k=1}^{m} Dk*Pre\_k_i" /></a>
 
* **AdaBoost正则化**  

     现在将回归问题和分类问题的最终的集成形式写为如下更为一般的形式
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=F_{m}(X)=\sum_{k=1}^{m}D_{k}*P_{k}(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{m}(X)=\sum_{k=1}^{m}D_{k}*P_{k}(X)" title="F_{m}(X)=\sum_{k=1}^{m}D_{k}*P_{k}(X)" /></a>
     
     也就是有：<a href="https://www.codecogs.com/eqnedit.php?latex=F_{k}(X)=&space;F_{k-1}(X)&space;&plus;&space;D_{k}*P_{k}(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{k}(X)=&space;F_{k-1}(X)&space;&plus;&space;D_{k}*P_{k}(X)" title="F_{k}(X)= F_{k-1}(X) + D_{k}*P_{k}(X)" /></a>
     
     现在将其正则化：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\\F_{k}(X)=&space;F_{k-1}(X)&space;&plus;&space;a&space;*D_{k}*P_{k}(X)\\&space;F_{0}(X)&space;=&space;\overrightarrow{\mathbf{0}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\F_{k}(X)=&space;F_{k-1}(X)&space;&plus;&space;a&space;*D_{k}*P_{k}(X)\\&space;F_{0}(X)&space;=&space;\overrightarrow{\mathbf{0}}" title="\\F_{k}(X)= F_{k-1}(X) + a *D_{k}*P_{k}(X)\\ F_{0}(X) = \overrightarrow{\mathbf{0}}" /></a>
     
     其中a为学习率，也就是步长。在调参时，和弱模型的最大个数一起调参。
     
   
* **AdaBoost答疑**   
   1. **这个弱模型可以是什么？**
   
        答：经常用的就是单层的决策树，也称为决策树桩(Decision Stump)，例如单层的CART。其实这个层数也是参数，需要交叉验证得到最好的。当然这个弱模型也可以是SVM、逻辑回归、神经网络等。
   
   2. **增加的样本权重如何在下一个模型训练时体现出作用？**
   
       答：以决策树为例，计算最佳分割点基尼系数时，或者MSE时，要乘以样本权重。总之，对于样本的计算量，要乘以相应的样本的权重。如果没在这个AdaBoost的框架下，则相当于原来的样本的权重都是一样的，所以乘或者不乘是一样的，现在在这个AdaBoost的框架下，因为样本权重发生了改变，所以需要乘。这样就把样本权重的改变体现了出来。
  
