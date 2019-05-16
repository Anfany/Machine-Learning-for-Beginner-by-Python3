# 卷积神经网络理论

### 一、基础

* **[图像结构](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/fig.md)**


* **卷积**

    * **[初识卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/convolution.md)**

    * **[再谈卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/convolution2.md)**

* **[池化](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/pooling.md)**

### 二、卷积神经网络

试想一下，针对图片这种形式的输入，如果依然用之前的全连接神经网络会如何。对于一个图片而言，假设其数字矩阵为320\*240\*3，如果使用全连接神经网络，也就是将数字矩阵变为长度为320\*240\*3=230400的向量，作为神经网络的输入。如果第一层的神经元的个数为100个，则全连接情形下第一层的参数就有230400\*100+100=23040100个，因此对于图片而言，全连接需要的参数过多。因此需要提取图片的多个特征，在保留这些特征的前提下，缩短可以表示这个图片的向量的长度。

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
     
     **1，输入层(INPUT)**：图片的数字矩阵为PM，维度为(90,117,3)，这一层的输出为PM；
     
     **2，卷积1层(CONV1)**：卷积核设为C1，C1_c表示第c个卷积核，其维度为(11,11,3)。这一层输出为C1_Out，维度为(30,39,96)；
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=C1\_Out&space;=&space;PM&space;*&space;C1&space;\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;C1\_Out[x,&space;y,&space;z]&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;=\sum_{i=1}^{3}&space;P\_M[x1,y1,i]&space;*&space;C1\_z[::i]&space;=&space;\sum_{i=1}^{3}\sum_{h=0}^{10}\sum_{s=0}^{10}Pm[i][h,s]&space;\times&space;Cm[i][h,s]\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;P\_M[x1,y1,i]\,\:&space;is\,&space;\:&space;the\,&space;\:&space;corrding&space;\,&space;\:&space;erea&space;\,&space;\:&space;of&space;PM\\&space;\\&space;.&space;\,&space;\,&space;\,&space;Pm[i]=P\_M[x1,y1,i],&space;Cm[i]=C1\_z[::i],\\&space;\\&space;.\,&space;\,&space;\,&space;x\in&space;[0,29],y\in&space;[0,38],z\in&space;[0,95]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C1\_Out&space;=&space;PM&space;*&space;C1&space;\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;C1\_Out[x,&space;y,&space;z]&space;\\&space;.\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;\,&space;=\sum_{i=1}^{3}&space;P\_M[x1,y1,i]&space;*&space;C1\_z[::i]&space;=&space;\sum_{i=1}^{3}\sum_{h=0}^{10}\sum_{s=0}^{10}Pm[i][h,s]&space;\times&space;Cm[i][h,s]\\&space;\\&space;.\,&space;\,&space;\,&space;\,&space;P\_M[x1,y1,i]\,\:&space;is\,&space;\:&space;the\,&space;\:&space;corrding&space;\,&space;\:&space;erea&space;\,&space;\:&space;of&space;PM\\&space;\\&space;.&space;\,&space;\,&space;\,&space;Pm[i]=P\_M[x1,y1,i],&space;Cm[i]=C1\_z[::i],\\&space;\\&space;.\,&space;\,&space;\,&space;x\in&space;[0,29],y\in&space;[0,38],z\in&space;[0,95]" title="C1\_Out = PM * C1 \\ \\ .\, \, \, \, \, \, C1\_Out[x, y, z] \\ .\, \, \, \, \, \, \, \, \, =\sum_{i=1}^{3} P\_M[x1,y1,i] * C1\_z[::i] = \sum_{i=1}^{3}\sum_{h=0}^{10}\sum_{s=0}^{10}Pm[i][h,s] \times Cm[i][h,s]\\ \\ .\, \, \, \, P\_M[x1,y1,i]\,\: is\, \: the\, \: corrding \, \: erea \, \: of PM\\ \\ . \, \, \, Pm[i]=P\_M[x1,y1,i], Cm[i]=C1\_z[::i],\\ \\ .\, \, \, x\in [0,29],y\in [0,38],z\in [0,95]" /></a>
     
     
     **3，激活函数1层(AF1)**：激活函数定义为Af1，输出为Af1_Out，维度为(30,39,96)；
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=Af1\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af1}(C1\_Out[x,y,z])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Af1\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af1}(C1\_Out[x,y,z])" title="Af1\_Out[x, y, z] = \mathbf{Af1}(C1\_Out[x,y,z])" /></a>
     
     **4，池化1层(POOL1)**：输出为Pool1_Out，维度为(10,13,96)；
     
      选择Af1_Out中的对应区域的最大值或者均值作为输出值。
     
     
     **5，卷积2层(CONV2)**：卷积核设为C2，C2_c表示第c个卷积核，其维度为(7,7,96)。这一层输出为C2_Out，维度为(5,7,256)；
     
        类似于卷积1层，此处不在赘述。
     
     **6，激活函数2层(AF2)**：激活函数定义为Af2，输出为Af2_Out，维度为(5,7,256)；
     
        <a href="https://www.codecogs.com/eqnedit.php?latex=Af2\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af2}(C2\_Out[x,y,z])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Af2\_Out[x,&space;y,&space;z]&space;=&space;\mathbf{Af2}(C2\_Out[x,y,z])" title="Af2\_Out[x, y, z] = \mathbf{Af2}(C2\_Out[x,y,z])" /></a>
     
     **7，池化2层(POOL2)**：输出为Pool2_Out，维度为(3,5,256)；
     
       选择Af2_Out中的对应区域的最大值或者均值作为输出值。
     
     **8，全连接1层(FC1)**：Pool2_Out变为一维的向量定义为In_Net，输入维度为(1,3840)，激活函数为FC1_af, 输出为FC1_Out，维度为(1,128)；
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=FC1\_Out&space;=&space;FC1\_af(\sum_{i=1}^{128}\sum_{j=1}^{3840}&space;W1[i,j]*In\_Net[j]&plus;B1[i])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FC1\_Out&space;=&space;FC1\_af(\sum_{i=1}^{128}\sum_{j=1}^{3840}&space;W1[i,j]*In\_Net[j]&plus;B1[i])" title="FC1\_Out = FC1\_af(\sum_{i=1}^{128}\sum_{j=1}^{3840} W1[i,j]*In\_Net[j]+B1[i])" /></a>
     
       其中W1为权重的矩阵，维度为(128,3840)，B2为阈值的矩阵，维度为(128,1)；
   
     **9，全连接2层(FC2)**：输入维度为(1,128)，激活函数为FC2_af, 输出为FC2_Out，维度为(1,4)；
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=FC2\_Out&space;=&space;FC2\_af(\sum_{i=1}^{4}\sum_{j=1}^{128}&space;W2[i,j]*FC1\_Out[j]&plus;B2[i])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FC2\_Out&space;=&space;FC2\_af(\sum_{i=1}^{4}\sum_{j=1}^{128}&space;W2[i,j]*FC1\_Out[j]&plus;B2[i])" title="FC2\_Out = FC2\_af(\sum_{i=1}^{4}\sum_{j=1}^{128} W2[i,j]*FC1\_Out[j]+B2[i])" /></a>
     
     其中W2为权重的矩阵，维度为(4,128)，B2为阈值的矩阵，维度为(4,1)；
     
     **10.输出层(OUTPUT)**：输入维度为(1,4)，输出为Net_Out，维度为(1,4)；
     
       <a href="https://www.codecogs.com/eqnedit.php?latex=Net\_Out&space;=\mathbf{&space;Softmax}(FC2\_Out)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Net\_Out&space;=\mathbf{&space;Softmax}(FC2\_Out)" title="Net\_Out =\mathbf{ Softmax}(FC2\_Out)" /></a>
     
     这个图片的标签为Real_Out，维度为(1,4)，元素代表图片的类别，属于此类别值为1，不属于此类别值为0。假设成本函数平方误差为代价函数，当然也可以为交叉熵代价函数。计算误差：
     
     <a href="https://www.codecogs.com/eqnedit.php?latex=\left\{\begin{matrix}&space;E&space;=&space;\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{4}(Real\_Out[i,j]-Net\_Out[i,j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N>1&space;\:&space;\:&space;\:&space;\:&space;(1)\\&space;\\&space;E&space;=&space;\sum_{j=1}^{4}(Real\_Out[j]-Net\_Out[j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N=1&space;\:&space;\:&space;\:&space;\:&space;(2)&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{\begin{matrix}&space;E&space;=&space;\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{4}(Real\_Out[i,j]-Net\_Out[i,j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N>1&space;\:&space;\:&space;\:&space;\:&space;(1)\\&space;\\&space;E&space;=&space;\sum_{j=1}^{4}(Real\_Out[j]-Net\_Out[j])^{2}&space;\:&space;\:&space;\:&space;\:&space;if&space;N=1&space;\:&space;\:&space;\:&space;\:&space;(2)&space;\end{matrix}\right." title="\left\{\begin{matrix} E = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{4}(Real\_Out[i,j]-Net\_Out[i,j])^{2} \: \: \: \: if N>1 \: \: \: \: (1)\\ \\ E = \sum_{j=1}^{4}(Real\_Out[j]-Net\_Out[j])^{2} \: \: \: \: if N=1 \: \: \: \: (2) \end{matrix}\right." /></a>
     
     本例中N=1，所以用式(1)。
     
    
   +  **卷积神经网络的反传播**
     
      获得误差后，就要进行反向传播，计算每个需要训练的参数的梯度。这里主要介绍全连接层、卷积层、池化层、激活层的反向传播，以及如何计算梯度和更新参数值。
     
      + **全连接层的反向传播**
      
      全连接层的反向传播和BP神经网络的一样，可以参考：[BP神经网络](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/BPNN/readme.md)。
       
      + **卷积层的反向传播**
      
      
      
      
       
      + **池化层的反向传播**
              
      + **激活层的反向传播**


   








