##  机器视觉数据集

   * **Mnist手写数字集**
      
      + [Mnist手写数字集下载以及说明](http://yann.lecun.com/exdb/mnist/)。四个.gz解压缩后可得到对应文件
      
            训练图片：train-images.idx3-ubyte；训练标签：train-labels.idx1-ubyte
            测试图片：t10k-images.idx3-ubyte；测试标签：t10k-labels.idx1-ubyte
      
      + Mnist手写数字集说明：
      
          所有图片均为28x28的灰度图片照片；训练数据集一共有6万张图片，测试数据集一共有1万张图片，合计7万张。数量如下所示：
         
         | 标签| 0|  1|  2|  3|  4|  5|  6|  7|  8|  9| 合计|
         |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
         | **训练**| 5,923|  6,742|5,958| 6,131|  5,842|  5,421|  5,918|  6,265|  5,851| 5,949|**60,000**|
         | **测试**| 980|  1,135|  1,032|  1,010|  982|  892|  958|  1,028|  974|  1,009|**10,000** |
         | **合计**| 6,903| 7,877| 6,990|7,141| 6,824| 6,313|  6,876|  7,293|  6,825|  6,958| **70,000**|
      
   * **Fashion-MNIST手写数据集**   
   
     + [Fashion-MNIST数据集下载以及说明](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.zh-CN.md)。四个.gz解压缩后可得到对应文件
 
            训练图片：train-images.idx3-ubyte；训练标签：train-labels.idx1-ubyte
            测试图片：t10k-images.idx3-ubyte；测试标签：t10k-labels.idx1-ubyte
            
      + Fashion-MNIST数据集说明：
            
          Fashion-MNIST是一个替代MNIST手写数字集的图像数据集，它是由Zalando(一家德国的时尚科技公司)旗下的研究部门提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。**Fashion-MNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致**。60000/10000的训练测试数据划分，28x28的灰度图片。  每个类别的训练、测试样本都是一样多的，分别为6000，1000个。
      
| 标签| 0|  1|  2|  3|  4|  5|  6|  7|  8|  9|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 对应</br>商品|T-shirt/Top</br>T恤| Trouser</br>裤子|Pullover</br>套衫|Dress</br>裙子|Coat</br>外套|Sandal</br>凉鞋|Shirt</br>汗衫| Sneaker</br>运动鞋|Bag</br>包|Ankle boot</br>踝靴|

