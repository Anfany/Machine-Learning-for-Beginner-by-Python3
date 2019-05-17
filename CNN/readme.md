# 卷积神经网络理论

### 一、基础

* **[图像结构](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/fig.md)**


* **卷积**

    * **[初识卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/convolution.md)**

    * **[再谈卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/convolution2.md)**

* **[池化](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/pooling.md)**

### 二、卷积神经网络

试想一下，针对图片这种形式的输入，如果依然用之前的全连接神经网络会如何。对于一个图片而言，假设其数字矩阵为320\*240\*3，如果使用全连接神经网络，也就是将数字矩阵变为长度为320\*240\*3=230400的向量，作为神经网络的输入，其中**320为高度，240是宽度，3为通道数或者深度**。如果第一层的神经元的个数为100个，则全连接情形下第一层的参数就有230400\*100+100=23040100个，因此对于图片而言，全连接需要的参数过多。因此需要提取图片的多个特征，在保留这些特征的前提下，缩短可以表示这个图片的向量的长度。

首先重申以下几点说明：图片其实就是数字矩阵；卷积就是提取图片中的特征；池化就是对特征中的信息进行下采样。卷积核是多维的，也就是分别对应输入的多个维度，输入和卷积核的对应维度的卷积的和加上该卷积核的阈值作为卷积的输出结果，因此得到的输出结果的维度就等于卷积核的个数。

下面开始介绍卷积神经网络的基本结构，这里只介绍最基本的形式，其他复杂构造的CNN单独介绍。



* **基本结构**

CNN一般是由输入层(INPUT)，卷积层(CONV)，激活层(AF)，池化层(POOL)，全连接层(FC)，输出层(OUTPUT)构成的。下面给出一个卷积神经网络的结构：
   
![卷积神经网络结构](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Cnn/cnn_s.png)
   

上面展示的结构仅仅是一个示例，在实际的应用中卷积层后面可以不设置激活层或者池化层，可以是多个卷积层直接相连，但是多个池化层相连的情况意义不大。最后的全连接层也可以是多个。构成模式可用如下表示：
```
输入层 ——> {{卷积层} ——> {激活层}？ ——> {池化层}？} ——> {全连接层} ——> 输出层
```
卷积、激活、池化的设置都比较自由，没有限制，但是都有比较好的经验可以借鉴。一个可以遵循的建议就是不要让需要调节的参数的数量骤降，要循序渐进的减少。下面以上面的示例说明卷积神经网络中需要调节的参数，也就是需要学习、需要训练的参数。
   
* **网络中的参数说明**   

上面示例结构中涉及到的需要训练的参数见下表，其中卷积核也是类似于神经元，也是有阈值的，当然也可以不设置。

![卷积神经网络结构](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Cnn/cnn_papng.png)
   
* **训练**      

和全连接神经网络一样，利用链式求导计算损失函数对每个权重的偏导数（梯度），然后根据梯度下降公式更新权重。整个过程分为两个过程，一是正向传播，二是反向传播。 下面以上面的示例详细说明卷积神经网络的训练过程，为了便于理解，以一张图片为例。
  

   + **卷积神经网络的正向传播**
     
     **1，输入层(INPUT)：图片的数字矩阵为PM，维度为(90,117,3)，这一层的输出为PM；**
     
     **2，卷积1层(CONV1)：卷积核设为C1，C1_c表示第c个卷积核，其维度为(11,11,3)。这一层输出为C1_Out，维度为(30,39,96)；**
     
        <a href="https://www.codecogs.com/eqnedit.php?latex=C1\_Out&space;=&space;PM&space;*&space;C1&space;\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;C1\_Out[x,&space;y,&space;z]&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;=\sum_{i=1}^{3}&space;P\_M[x1,y1,i]&space;*&space;C1\_z[::i]&space;=&space;\sum_{i=1}^{3}\sum_{h=0}^{10}\sum_{s=0}^{10}Pm[i][h,s]&space;\times&space;Cm[i][h,s]\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;P\_M[x1,y1,i]\,\:&space;is\,&space;\:&space;the\,&space;\:&space;corrding&space;\,&space;\:&space;erea&space;\,&space;\:&space;of&space;PM\\&space;\\&space;.&space;\,&space;\,&space;\,&space;Pm[i]=P\_M[x1,y1,i],&space;Cm[i]=C1\_z[::i],\\&space;\\&space;.\,&space;\,&space;\,&space;x\in&space;[0,29],y\in&space;[0,38],z\in&space;[0,95]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C1\_Out&space;=&space;PM&space;*&space;C1&space;\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;C1\_Out[x,&space;y,&space;z]&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;=\sum_{i=1}^{3}&space;P\_M[x1,y1,i]&space;*&space;C1\_z[::i]&space;=&space;\sum_{i=1}^{3}\sum_{h=0}^{10}\sum_{s=0}^{10}Pm[i][h,s]&space;\times&space;Cm[i][h,s]\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;P\_M[x1,y1,i]\,\:&space;is\,&space;\:&space;the\,&space;\:&space;corrding&space;\,&space;\:&space;erea&space;\,&space;\:&space;of&space;PM\\&space;\\&space;.&space;\,&space;\,&space;\,&space;Pm[i]=P\_M[x1,y1,i],&space;Cm[i]=C1\_z[::i],\\&space;\\&space;.\,&space;\,&space;\,&space;x\in&space;[0,29],y\in&space;[0,38],z\in&space;[0,95]" title="C1\_Out = PM * C1 \\ \\ .\, \, \, \, \, \, C1\_Out[x, y, z] \\ .\, \, \, \, \, \, \, \, \, =\sum_{i=1}^{3} P\_M[x1,y1,i] * C1\_z[::i] = \sum_{i=1}^{3}\sum_{h=0}^{10}\sum_{s=0}^{10}Pm[i][h,s] \times Cm[i][h,s]\\ \\ .\, \, \, \, P\_M[x1,y1,i]\,\: is\, \: the\, \: corrding \, \: erea \, \: of PM\\ \\ . \, \, \, Pm[i]=P\_M[x1,y1,i], Cm[i]=C1\_z[::i],\\ \\ .\, \, \, x\in [0,29],y\in [0,38],z\in [0,95]" /></a>
     
     
     **3，激活函数1层(AF1)：激活函数定义为Af1，输出为Af1_Out，维度为(30,39,96)；**
     
        <a href="https://www.codecogs.com/eqnedit.php?latex=Af1\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af1}(C1\_Out[x,y,z])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Af1\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af1}(C1\_Out[x,y,z])" title="Af1\_Out[x, y, z] = \mathbf{Af1}(C1\_Out[x,y,z])" /></a>
     
     **4，池化1层(POOL1)：输出为Pool1_Out，维度为(10,13,96)；**
     
          
        选择Af1_Out中的对应区域的最大值或者均值作为输出值。
     
     
     **5，卷积2层(CONV2)：卷积核设为C2，C2_c表示第c个卷积核，其维度为(7,7,96)。这一层输出为C2_Out，维度为(5,7,256)；**
     
        类似于卷积1层，此处不在赘述。
   
     **6，激活函数2层(AF2)：激活函数定义为Af2，输出为Af2_Out，维度为(5,7,256)；**
     
        <a href="https://www.codecogs.com/eqnedit.php?latex=Af2\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af2}(C2\_Out[x,y,z])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Af2\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af2}(C2\_Out[x,y,z])" title="Af2\_Out[x, y, z] = \mathbf{Af2}(C2\_Out[x,y,z])" /></a>
     
     **7，池化2层(POOL2)：输出为Pool2_Out，维度为(3,5,256)；**
     
        选择Af2_Out中的对应区域的最大值或者均值作为输出值。
     
     **8，全连接1层(FC1)：Pool2_Out变为一维的向量定义为In_Net，输入维度为(1,3840)，激活函数为FC1_af, 输出为FC1_Out，维度为(1,128)；**
     
       <a href="https://www.codecogs.com/eqnedit.php?latex=FC1\_Out[i]&space;=&space;FC1\_af(\sum_{j=0}^{3839}&space;W1[i,j]*In\_Net[j]&plus;B1[i]),i=0,1,\cdots&space;127" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FC1\_Out[i]&space;=&space;FC1\_af(\sum_{j=0}^{3839}&space;W1[i,j]*In\_Net[j]&plus;B1[i]),i=0,1,\cdots&space;127" title="FC1\_Out[i] = FC1\_af(\sum_{j=0}^{3839} W1[i,j]*In\_Net[j]+B1[i]),i=0,1,\cdots 127" /></a>
     
       其中W1为权重的矩阵，维度为(128,3840)，B1为阈值的矩阵，维度为(128,1)；
   
     **9，全连接2层(FC2)：输入维度为(1,128)，激活函数为FC2_af, 输出为FC2_Out，维度为(1,4)；**
     
       <a href="https://www.codecogs.com/eqnedit.php?latex=FC2\_Out[i]&space;=&space;FC2\_af(\sum_{j=0}^{127}&space;W2[i,j]*FC1\_Out[j]&plus;B2[i]),i=0,1,2,3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FC2\_Out[i]&space;=&space;FC2\_af(\sum_{j=0}^{127}&space;W2[i,j]*FC1\_Out[j]&plus;B2[i]),i=0,1,2,3" title="FC2\_Out[i] = FC2\_af(\sum_{j=0}^{127} W2[i,j]*FC1\_Out[j]+B2[i]),i=0,1,2,3" /></a>
   
     
     其中W2为权重的矩阵，维度为(4,128)，B2为阈值的矩阵，维度为(4,1)；
     
     **10.输出层(OUTPUT)：输入维度为(1,4)，输出为Net_Out，维度为(1,4)；**
     
       <a href="https://www.codecogs.com/eqnedit.php?latex=Net\_Out&space;=\mathbf{&space;Softmax}(FC2\_Out)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Net\_Out&space;=\mathbf{&space;Softmax}(FC2\_Out)" title="Net\_Out =\mathbf{ Softmax}(FC2\_Out)" /></a>
     
       这个图片的标签为Real_Out，维度为(1,4)，元素代表图片的类别，属于此类别值为1，不属于此类别值为0。假设成本函数平方误差为代价函数，当然也可以为交叉熵代价函数。计算误差：
     
      <a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;E&space;=&space;\frac{1}{2N}\sum_{i=1}^{N}\sum_{j=1}^{4}(Real\_Out[i,j]-Net\_Out[i,j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N>1&space;\:&space;\:&space;\:&space;\:&space;(1)\\&space;\\&space;E&space;=&space;\frac{1}{2}\sum_{j=1}^{4}(Real\_Out[j]-Net\_Out[j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N=1&space;\:&space;\:&space;\:&space;\:&space;(2)&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;E&space;=&space;\frac{1}{2N}\sum_{i=1}^{N}\sum_{j=1}^{4}(Real\_Out[i,j]-Net\_Out[i,j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N>1&space;\:&space;\:&space;\:&space;\:&space;(1)\\&space;\\&space;E&space;=&space;\frac{1}{2}\sum_{j=1}^{4}(Real\_Out[j]-Net\_Out[j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N=1&space;\:&space;\:&space;\:&space;\:&space;(2)&space;\end{matrix}\right." title="\left\{\begin{matrix} E = \frac{1}{2N}\sum_{i=1}^{N}\sum_{j=1}^{4}(Real\_Out[i,j]-Net\_Out[i,j])^{2} \: \: \: \: if N>1 \: \: \: \: (1)\\ \\ E = \frac{1}{2}\sum_{j=1}^{4}(Real\_Out[j]-Net\_Out[j])^{2} \: \: \: \: if N=1 \: \: \: \: (2) \end{matrix}\right." /></a>
     
     本例中N=1，所以用式(2)。
     
    
   +  **卷积神经网络的反传播**
     
      获得误差E后，进行反向传播，计算每个需要训练的参数的梯度。需要训练的参数在全连接层以及卷积层中。在池化和激活层中，不存在需要训练的参数，因此经过这些层时，只要把梯度传递到前一层即可。详见下面的叙述。
     
         + 首先计算成本函数E对全连接层FC2、FC1中的权重个阈值的梯度，这个和[BP神经网络](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/BPNN/readme.md)的计算是一样的，此处不在赘述。
      
         + 计算成本函数E对于**池化层POOL2**中的梯度。因为池化层没有需要训练的参数，因此只需要传递梯度即可。假设成本函数E对于In_Net的梯度向量为d8，其维度为(1,3840)，现在将其变为维度为(3,5,256)的梯度矩阵dm8。下面分2种情况说明梯度矩阵如何传递：
           
             1. 如果池化是最大值池化：对于dm8[a,b,c]而言，就是将dm8[a,b,c]所对应的Af2_Out中的矩阵块中，具有最大值的位置的梯度设为dm8[a,b,c]，其他的设置为0。
             
             2. 如果池化是均值池化：对于dm8[a,b,c]而言，就是将dm8[a,b,c]所对应的Af2_Out中的矩阵块中，所有的位置的梯度设置为dm8[a,b,c]除以矩阵块中元素的个数。如果对于有重叠的池化，则具有多个梯度值的可以计算梯度的和作为该位置的最终的梯度。
             
            此时得到的梯度矩阵d7的维度应该是和Af2_Out的维度是一样的。
             
         
        +  计算成本函数E对于**激活层AF2**中的梯度。因为激活层也没有训练的参数，因此只需要传递矩阵即可。根据上面得到的梯度矩阵d7，假设这一层得到的梯度矩阵为d6，则有：
            
            <a href="https://www.codecogs.com/eqnedit.php?latex=d6[x,y,z]=d7[x,y,z]*&space;{Af2}'(C2\_Out[x,y,z])\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;{Af2}'(s)=\frac{\partial&space;Af2(s)}{\partial&space;s}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d6[x,y,z]=d7[x,y,z]*&space;{Af2}'(C2\_Out[x,y,z])\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;{Af2}'(s)=\frac{\partial&space;Af2(s)}{\partial&space;s}" title="d6[x,y,z]=d7[x,y,z]* {Af2}'(C2\_Out[x,y,z])\\ \\ .\, \, \, \, \, \, \, {Af2}'(s)=\frac{\partial Af2(s)}{\partial s}" /></a>
     
            也就是说，传递过来的梯度值d7[x,y,z]与激活函数的导数在点C2_Out[x,y,z]处的值的乘积就是位置[x,y,z]的梯度d6[x,y,z]。
       
       + 下面计算成本函数E对于**卷积层CONV2**的梯度矩阵，因为卷积层中需要训练的参数就是所有卷积核矩阵中的数。现在重申下符号说明，卷积层CONV2的输入为Pool1_Out，维度为(10,13,96)，卷积核的个数为K=256，单个卷积核的维度为(7,7,96)，输出为C2_Out，维度为(5,7,256)。具体运算可参考下图，偏置的个数为输入矩阵的维度乘以卷积核的个数：
       
          ![卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Cnn/cnn_conv.png)
          
            <a href="https://www.codecogs.com/eqnedit.php?latex=C2\_Out=Pool1\_Out&space;*&space;C2,&space;\:&space;\:&space;\:&space;\:&space;\:&space;a*b&space;\:&space;\:&space;\:is&space;\:&space;\:&space;\:&space;convolution&space;\:&space;\:&space;\:&space;between&space;\:&space;\:&space;\:&space;a&space;\:&space;\:&space;\:&space;and&space;\:&space;\:&space;\:&space;b\\&space;\\&space;.\:&space;\:&space;\:&space;\:&space;\:&space;d5&space;=&space;\frac{\partial&space;E}{\partial&space;C2}=&space;\frac{\partial&space;E}{\partial&space;C2\_Out}&space;\times&space;\frac{\partial&space;C2\_Out}{\partial&space;C2}=d6&space;\times&space;\frac{\partial&space;C2\_Out}{\partial&space;C2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C2\_Out=Pool1\_Out&space;*&space;C2,&space;\:&space;\:&space;\:&space;\:&space;\:&space;a*b&space;\:&space;\:&space;\:is&space;\:&space;\:&space;\:&space;convolution&space;\:&space;\:&space;\:&space;between&space;\:&space;\:&space;\:&space;a&space;\:&space;\:&space;\:&space;and&space;\:&space;\:&space;\:&space;b\\&space;\\&space;.\:&space;\:&space;\:&space;\:&space;\:&space;d5&space;=&space;\frac{\partial&space;E}{\partial&space;C2}=&space;\frac{\partial&space;E}{\partial&space;C2\_Out}&space;\times&space;\frac{\partial&space;C2\_Out}{\partial&space;C2}=d6&space;\times&space;\frac{\partial&space;C2\_Out}{\partial&space;C2}" title="C2\_Out=Pool1\_Out * C2, \: \: \: \: \: a*b \: \: \:is \: \: \: convolution \: \: \: between \: \: \: a \: \: \: and \: \: \: b\\ \\ .\: \: \: \: \: d5 = \frac{\partial E}{\partial C2}= \frac{\partial E}{\partial C2\_Out} \times \frac{\partial C2\_Out}{\partial C2}=d6 \times \frac{\partial C2\_Out}{\partial C2}" /></a>
            
         
            
            其中上一层得到的梯度矩阵为d6，其维度为(5,7,256)。定义这一层的梯度矩阵为d5，其维度应该为(7,7,96,256)。上面式子给出了大致的关系，但是具体的维度对应还需要进一步讨论：
            
            
            
            
            
          
       
       
      
      
      
      
      


   








