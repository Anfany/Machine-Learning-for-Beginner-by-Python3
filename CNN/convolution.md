# 初识卷积
#### 一、定义

卷积，和加减乘除一样，是一种数学运算。下面给出它的定义：f，g的卷积记为(f\*g)，其中：

* **连续情形：** <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{\color{Blue}&space;(f*g)(t)&space;=&space;\int_{a}^{b}&space;f(\tau)&space;g(t&space;-&space;\tau&space;)\mathit{d}\tau&space;}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{\color{Blue}&space;(f*g)(t)&space;=&space;\int_{a}^{b}&space;f(\tau)&space;g(t&space;-&space;\tau&space;)\mathit{d}\tau&space;}}" title="\mathbf{{\color{Blue} (f*g)(t) = \int_{a}^{b} f(\tau) g(t - \tau )\mathit{d}\tau }}" /></a>

* **离散情形：** <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Red}&space;(f&space;*&space;g)(x)&space;=&space;\sum_{\tau&space;=&space;a}^{b}&space;f(\tau)g(x-\tau)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{Red}&space;(f&space;*&space;g)(x)&space;=&space;\sum_{\tau&space;=&space;a}^{b}&space;f(\tau)g(x-\tau)}" title="{\color{Red} (f * g)(x) = \sum_{\tau = a}^{b} f(\tau)g(x-\tau)}" /></a>

其中[a, b]为函数的定义域，连续情形下f(x), g(x)在定义域区间内是可积的。

#### 二、示例：高利贷利息

  
  假设賴某每月都向某机构贷款f(t)元，贷款的利息是按复利计算，月利率3%。计算N个月月底賴某需要付出的利息**P(N)**？
  
  
  ![复利](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/c1.png)
  
  
  利息
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{P(N)}&space;=&space;{\color{DarkOrange}&space;f(N)*3\%&space;}&plus;&space;\mathbf{f(N-1)*(1&plus;3\%)*3\%&space;}&plus;&space;\cdots&space;&plus;&space;{\color{Red}&space;\mathbf{f(1)}*(1&plus;3\%)^{(N-1)}*3\%}&space;\\&space;\\&space;=&space;{\color{DarkOrange}&space;f(N)*&space;g(0))}&plus;&space;\mathbf{f(N-1)*g(1)}&plus;&space;\cdots&space;&plus;&space;{\color{Red}&space;\mathbf{f(1)}*g(N-1)}&space;\\&space;\\&space;=&space;{\color{DarkOrange}&space;f(N)*&space;g(N&space;-&space;N))}&plus;&space;\mathbf{f(N-1)*g(N&space;-&space;(N-1))}&plus;&space;\cdots&space;&plus;&space;{\color{Red}&space;\mathbf{f(1)}*g(N-1)}&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\mathbf{P(N)}&space;=&space;{\color{DarkOrange}&space;f(N)*3\%&space;}&plus;&space;\mathbf{f(N-1)*(1&plus;3\%)*3\%&space;}&plus;&space;\cdots&space;&plus;&space;{\color{Red}&space;\mathbf{f(1)}*(1&plus;3\%)^{(N-1)}*3\%}&space;\\&space;\\&space;=&space;{\color{DarkOrange}&space;f(N)*&space;g(0))}&plus;&space;\mathbf{f(N-1)*g(1)}&plus;&space;\cdots&space;&plus;&space;{\color{Red}&space;\mathbf{f(1)}*g(N-1)}&space;\\&space;\\&space;=&space;{\color{DarkOrange}&space;f(N)*&space;g(N&space;-&space;N))}&plus;&space;\mathbf{f(N-1)*g(N&space;-&space;(N-1))}&plus;&space;\cdots&space;&plus;&space;{\color{Red}&space;\mathbf{f(1)}*g(N-1)}&space;\\" title="\\ \mathbf{P(N)} = {\color{DarkOrange} f(N)*3\% }+ \mathbf{f(N-1)*(1+3\%)*3\% }+ \cdots + {\color{Red} \mathbf{f(1)}*(1+3\%)^{(N-1)}*3\%} \\ \\ = {\color{DarkOrange} f(N)* g(0))}+ \mathbf{f(N-1)*g(1)}+ \cdots + {\color{Red} \mathbf{f(1)}*g(N-1)} \\ \\ = {\color{DarkOrange} f(N)* g(N - N))}+ \mathbf{f(N-1)*g(N - (N-1))}+ \cdots + {\color{Red} \mathbf{f(1)}*g(N-1)} \\" /></a>
  
  其中<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{g(x)&space;=&space;(1&plus;3\%)^{x}*3\%}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{g(x)&space;=&space;(1&plus;3\%)^{x}*3\%}" title="\mathbf{g(x) = (1+3\%)^{x}*3\%}" /></a>
  
* **离散情形：**
  
此时利息的公式为：<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{P(N)}&space;=&space;\mathbf{\sum_{\tau&space;=&space;1}^{N}f(\tau)*g(N-\tau&space;)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{P(N)}&space;=&space;\mathbf{\sum_{\tau&space;=&space;1}^{N}f(\tau)*g(N-\tau&space;)}" title="\mathbf{P(N)} = \mathbf{\sum_{\tau = 1}^{N}f(\tau)*g(N-\tau )}" /></a>
  

* **连续情形：**
  
将借款的时间间隔无限缩小，利息的计算尺度也相应的缩小。问题就可以转变为连续情形，此时利息的公式为：<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{P(N)}&space;=&space;\mathbf{\int_{\tau&space;=&space;1}^{N}f(\tau)*g(N-\tau&space;)}\mathit{\mathbf{d}\tau}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{P(N)}&space;=&space;\mathbf{\int_{\tau&space;=&space;1}^{N}f(\tau)*g(N-\tau&space;)}\mathit{\mathbf{d}\tau}" title="\mathbf{P(N)} = \mathbf{\int_{\tau = 1}^{N}f(\tau)*g(N-\tau )}\mathit{\mathbf{d}\tau}" /></a>
  
将上面的示例抽象表示，借款好比输入，计算利息的方式可看作一个系统，利息的多少可看作输出。也就是输出等于输入与系统的卷积。设置不同的系统，就可以得到不同的输出。 卷积的应用很多，下面主要介绍在图像方面的应用。

#### 三、图像卷积

图像可以看成一个三维矩阵，通过卷积的方式就可以得到图像中的特征信息。图像是输入，不同的卷积核(又称过滤器)可看作不同的系统，图像和卷积核经过卷积得到的结果就可看作图像中的特征信息。设置不同的卷积核，可以对图像进行不同的处理，也就是获得图像的不同的特征信息。下面首先以单通道的数字矩阵的卷积为例说明。[卷积程序参见](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/convolution.py)。


* **卷积核说明**
  
卷积核一般又称为过滤器，卷积核就是一个数字矩阵，一般设置为奇数行奇数列。因为在计算机视觉里，如果有一个中心像素点会更方便，因此卷积核数字矩阵行列均为同一个奇数。在卷积神经网络中卷积核通常设置为1\*1，3\*3，5\*5，7\*7。此外，需要注意的是，在卷积神经网络中，卷积核一般不需要进行镜像旋转。


* **卷积操作**

下面介绍单通道的数字矩阵的卷积是如何操作的：
  
![valid卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/c_valid.png)
   
  
   + **Padding填充**
     
   从上例可以看出，一个N\*M的矩阵与一个F\*F的卷积核的卷积结果为(N-F+1)\*(M-F+1)的矩阵。矩阵的行和列都减小了。这种卷积方式就是Valid卷积。在卷积神经网络中因为有多个卷积层，这样会导致图片越来越小，也就是会损失一些图像的信息，因此为了保证图片原来的维度，需要进行Padding，也就是填充。填充主要有4种形式：补零填充，边界复制填充，镜像填充，块填充。本文主要介绍常用的补零填充，就是在图片的数字矩阵的四周添加上值为0的网格。
  
   + **Valid卷积**     
     
   示例中的就是Valid卷积，也就是Padding为0。
  
   + **Same卷积**
     
   经过Same卷积后，矩阵的行和列均不变，此时就需要Padding，填充的行和列数分别为F-1，F-1，其中F为卷积核矩阵的行和列数。也就是在图片数字矩阵的上、下均添加(F-1)/2行元素为0的网格，左、右均添加(F-1)/2列元素为0的网格。具体参加下图：
    
   ![same卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/c_same.png)
   
   + **Full卷积**
     
   Full卷积就是在数字矩阵的四周填充F-1行，F-1列的值为0的网格。卷积后得到的结果为N+F-1行，M+F-1列。具体参见下图：
   
   ![full卷积](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/c_full.png) 
     
* **Strided步长**
   
步长就是上面的黄色部分每一次移动的步伐的长度。上面示例中显示的两种方式的卷积的步长s均为1，下面图示给出移动步长s为2，3的情况：
   
   
![Strided步长](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/cs.png)
   
  
根据上图可知，步长的不同也会最终影响卷积结果的维度。对于N\*M的矩阵，卷积核矩阵为F\*F，步长为s，填充Padding为P，那么卷积后得到的结果的行row、列column分别为：
   
![维度计算](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/c_rc.png)
   
   
* **卷积结果中数字的处理**
  
对于卷积结果中的特殊数字，例如小于0大于255的数字，以及[0, 255]区间内的小数，因为在卷积神经网络中，卷积层后面一般会跟着激活层，因此不用对这些数字特别处理。如果要把卷积结果进行显示的话，可以把这些数字变为unit8的形式。
    
* **不同卷积核的对比**
  
正如前文提到的，不同的卷积核就相当于不同的图像特征提取器。至于为何这种形式的卷积核，就可以提取这种特征，本文不再多做说明。下文给出不同的卷积核得到的图像不同的特征的对比。也就是图像的三个通道的数字矩阵均用同一个卷积核进行same卷积，然后将各自通道的卷积结果合成的的图片进行对比。下面的示例图片是图像处理最为经典的图片，图片中的人为Lena。
    

| 卷积核名称 | 卷积核 | 功能| 图片显示 | 
| :------:| :------: | :------: | :------: |
| 单位卷积核| **0  0  0 <br>0  1 0<br>0 0 0** | 原图 |  ![原始图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/lena.jpg)|
| 均值卷积核| **1/9  1/9  1/9 <br>1/9 1/9 1/9<br>1/9 1/9 1/9** | 模糊 |  ![均值模糊图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/lena_means.png)|
| 高斯模糊卷积核| ![高斯模糊](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/gauss.png) | 高斯模糊 | ![高斯模糊图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/lena_gauss.png)| 
| 拉普拉斯卷积核| **-1 -1 -1 <br>-1 8 -1<br>-1 -1 -1** | 边缘检测、锐化|  ![锐化图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Convolution/lena_laplace.png)|   
  
  
  
  
  
  
  
  
  
  
  
  
