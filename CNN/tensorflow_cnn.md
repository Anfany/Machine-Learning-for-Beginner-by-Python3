## TensorFlow(V1.13.1)中CNN的函数说明

* **卷积**

#### tf.nn.conv2d：根据4维的矩阵和4维的卷积核计算2维的卷积

```python
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)
```


**input**：需要进行卷积的数字矩阵。一个四维的Tensor，数据类型可为```half```，```bfloat16```，```float32```或者```float64```。维度的排列顺序和data_format的设置有关；

**filter**：代表卷积核的数字矩阵。一个四维的Tensor。4维的维度的排列顺序为[高度，宽度，通道数\深度，卷积核的个数]；

**strides**：卷积时图像每个维度的步长。一个长度为4的一维整型向量，；

**padding**：卷积的方式。字符串类型，"SAME"或者"VALID"；

**use_cudnn_on_gpu**：是否使用cudnn加速。布尔类型，默认为True；

**data_format**：输入数据和输出数据的维度的排列形式。字符串类型，"NHWC"或者"NCHW"，默认为前者，此时数据维度的排列顺序为[批训练的样本数，高度，宽度，通道数\深度]。当为"NCHW"时，数据维度的排列顺序为[批训练的样本数，通道数\深度，高度，宽度]；

**dilations**：对应输入矩阵维度的扩张因子列表。一个长度为4的一维整型向量。默认值为[1，1，1，1]。对应的维度的顺序为data_format确定的维度顺序，其中批训练的样本数和通道数\深度对应的值必须为1；

**name**：这个算子的名称。属于可选参数；


* **池化**

   #### tf.nn.avg_pool：均值池化
   
   ```python
   tf.nn.avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
   ```
   
   #### tf.nn.max_pool：最大值池化
   
      
   ```python
   tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
   ```
   
   
   
