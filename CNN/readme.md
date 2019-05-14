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
   

上面展示的结构仅仅是一个示例，在实际的应用中卷积层后面可以不设置激活层或者池化层，可以是多个卷积层直接相连，但是多个池化层相连的情况很罕见。最后的全连接层也可以是多个。卷积、激活、池化的设置都比较自由，没有限制，但是都有比较好的经验可以借鉴。一个可以遵循的建议就是不要让需要调节的参数的数量骤降，要循序渐进的减少。下面以上面的示例说明卷积神经网络中需要调节的参数，也就是需要学习、需要训练的参数。
   
* **网络中的参数说明**   

上面示例结构中涉及到的需要训练的参数见下表，其中卷积核也是类似于神经元，也是有阈值的，当然也可以不设置。

![卷积神经网络结构](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Cnn/cnn_p.png)
   
* **训练**      



   








