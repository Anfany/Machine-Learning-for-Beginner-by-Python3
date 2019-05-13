# 卷积神经网络理论

### 一、基础

* **[图像结构](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/fig.md)**


* **卷积**

    * **[初识卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/convolution.md)**

    * **[再谈卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/convolution2.md)**

* **[池化](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/pooling.md)**

### 二、卷积神经网络

下面三点已经知道：图片其实就是数字矩阵；卷积就是提取图片中的特征；池化就是对特征中的信息进行采样。下面开始介绍卷积神经网络的基本结构，这里只介绍最基本的形式，至于其他复杂的形式单独介绍。试想一下，如果依然用之前的全连接神经网络会如何。对于一个图片而言，假设其数字矩阵为320\*240\*3，如果使用全连接神经网络，也就是将数字矩阵变为长度为320\*240\*3=230400的向量，作为一个图片的输入，如果第一层的神经元的个数为100个，则全连接情形下参数就等于230400\*100+100=23040100个，因此对于图片而言，全连接需要的参数过多。因此需要提取图片的特征，在保留特征的前提下，缩短可以表示这个图片的向量的长度。


* **基本结构**

   CNN一般是由输入层(INPUT)，卷积层(CONV)，激活层(AF)，池化层(POOL)，全连接层(FC)，输出层(OUTPUT)构成的。下面给出一个卷积神经网络的结构：
   
   
   
   
   
   
* **参数说明**   
   
   
   
* **训练**      



   








