# 池化

池化(Pooing)操作的对象是单通道的数字矩阵，也就是对该矩阵某一个邻域内的数字集合进行采样。主要有3种形式：一般池化，重叠池化和金字塔池化。 

### 一、池化类型

[程序参见](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/pooling.py)

* **一般池化**
   
  池化窗口的尺寸为n\*n，一般情况下池化窗口都是正方形的。步长等于n。此时池化窗口之间是没有重叠的。对于超出数字矩阵范围的，只计算范围内的或者范围外的用0填充在计算。本文只介绍最大值池化，均值池化，随机池化。下面给出图示：

  + **最大值池化(Max Pooling)**
  
  池化窗口范围内的最大值作为采样的输出值。
  
  ![最大值池化](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/max_pool.png)
  
  + **均值池化(Average pooling)**
  
  池化窗口范围内的平均值作为采样的输出值，也就是普通均值池化。或者将范围内的数字归一化，每个数字与该范围内的数字之和的比例作为该数字的权重，然后原始数字和对应权重的乘积的和作为最终的输出值，也就是加权平均。下图中的示例是前者。
  
  ![均值池化](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/mean_pool.png)
  
  + **随机池化(Stochastic pooling)**
  
  池化窗口范围内的数字，采用轮盘赌的方式进行采样，就是值较大的数字成为采样输出值的概率较大。 
  
  
* **重叠池化**  

 池化窗口之间有重叠。也就是步长大于等于1小于n，计算和一般池化是一样的。

* **Spatial Pyramid Pooling 空间金字塔池化**  

  
  
  
### 二、池化作用

   1. 降低了图片尺寸，也就是增大了感受野。感受野就是数字矩阵中的一个数字所对应的原图中的区域大小。因为池化是在某个范围内选择一个数字，也就是让这个数字代表这个范围内的所有的像素得值。这样做虽然也丢失了一些图片信息，但是同时增加了鲁棒性。
   
   2. 增加平移不变性。图片中某个目标单纯的位置的移动，不应该影响识别结果。而池化捕捉的恰好是目标的特征，并不是目标所在的位置，因此增加了平移不变性。
   
   3、提升训练速度。因为在保留特征信息的前提下，降低了图片的尺寸。

### 三、池化对比

![原始图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/lena.jpg)


| 池化方法 | 池化 | 图片显示 | 
| :------:|:------:|:------:|
| 一般：最大值| **尺寸=8，步长=10** | ![一般最大值图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/max_normal.png)|
| 重叠：最大值| **尺寸=8，步长=4**| ![重叠最大值图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/max.png)|
| 一般：普通均值| **尺寸=8，步长=10** | ![普通图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/mean_normal.png)|
| 一般：加权均值| **尺寸=8，步长=10** | ![加权图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/mean_weight.png)| 

  
[程序参见](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/Pooling/pooling_fig.py)


