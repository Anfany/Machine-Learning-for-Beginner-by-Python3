
### 图像的结构

图像是由一个个像素点构成，每个像素点有三个通道，分别对应R，G，B。彩色RGB图像其实是一个三维矩阵，矩阵中的每个数字(0到255)代表的是一个像素一个通道的灰度。下面举例说明：

![彩色图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af.png)![图片描述](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af_doc.png)![像素图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af_pixel.png)

#### 一、读取这个图片的三维矩阵
```python
# -*- coding：utf-8 -*-
# &Author  AnFany

from skimage import io

fig_path = r"C:\Users\GWT9\Desktop\af.png"
matrix = io.imread(fig_path)  
print(matrix, matrix.shape)
```

* 结果
```python
[[[137 198 182]
  [126 189 172]
  [112 178 164]
  ...
  [106 220 210]
  [ 94 208 198]
  [ 96 210 202]]
  
...

 [[210 227 211]
  [199 216 200]
  [213 227 214]
  ...
  [142 204 193]
  [147 207 197]
  [151 211 199]]] (189, 149, 3)
```

可以得出图片的数字矩阵维度为(189, 149, 3)。其中第一行第一列的像素(上图右)对应的RGB是```[137 198 182]```，第一行第二列的像素对应的RGB是```[126 189 172]```；最后一行第一列像素对应的就是```[210 227 211]```。

#### 二、获取3个通道的矩阵以及维度

```python
# -*- coding：utf-8 -*-
# &Author  AnFany

from skimage import io
from PIL import Image


fig_path = r"C:\Users\GWT9\Desktop\af.png"   # 图片路径
matrix = io.imread(fig_path)


R = matrix[:, :, 0]  # 获取R通道
G = matrix[:, :, 1]  # 获取G通道
B = matrix[:, :, 2]  # 获取B通道

print('R通道矩阵：\n', R, '维度：', R.shape)
print('G通道矩阵：\n', G, '维度：', G.shape)
print('B通道矩阵：\n', B, '维度：', B.shape)
```
* 结果
```python
R通道矩阵：
 [[137 126 112 ... 106  94  96]
 [153 171 154 ... 114 100 116]
 [144 137 137 ... 112 106 125]
 ...
 [218 216 206 ... 136 140 140]
 [221 210 211 ... 139 143 145]
 [210 199 213 ... 142 147 151]] 维度： (189, 149)
G通道矩阵：
 [[198 189 178 ... 220 208 210]
 [214 233 220 ... 226 212 227]
 [206 199 201 ... 220 214 235]
 ...
 [232 230 222 ... 202 204 203]
 [235 224 227 ... 203 205 207]
 [227 216 227 ... 204 207 211]] 维度： (189, 149)
B通道矩阵：
 [[182 172 164 ... 210 198 202]
 [198 218 206 ... 214 200 218]
 [191 184 187 ... 207 201 224]
 ...
 [219 217 211 ... 188 190 192]
 [222 211 216 ... 189 192 194]
 [211 200 214 ... 193 197 199]] 维度： (189, 149)
```

#### 三、根据每个通道的数字矩阵输出图片

```python
# -*- coding：utf-8 -*-
# &Author  AnFany

from skimage import io
from PIL import Image


fig_path = r"C:\Users\GWT9\Desktop\af.png"   # 图片路径
matrix = io.imread(fig_path)


R = matrix[:, :, 0]  # 获取R通道
G = matrix[:, :, 1]  # 获取G通道
B = matrix[:, :, 2]  # 获取B通道

R_image = Image.fromarray(R)
R_image.show()
R_image.save(r"C:\Users\GWT9\Desktop\af_R.png")


G_image = Image.fromarray(G)
G_image.show()
G_image.save(r"C:\Users\GWT9\Desktop\af_G.png")


B_image = Image.fromarray(B)
B_image.show()
B_image.save(r"C:\Users\GWT9\Desktop\af_B.png")
```

* 结果

  + **通道R**
  
  ![通道R](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af_R.png)!
  
  + **通道G**
  
  ![通道G](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af_G.png)!
  

  + **通道B**

  ![通道B](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af_B.png)!


#### 四、生成自己的专属图片
```python
# -*- coding：utf-8 -*-
# &Author  AnFany

# -*- coding：utf-8 -*-
# &Author  AnFany


#  根据生肖数+星座数+出生时辰+幸运数字生成专属图片

from PIL import Image   # 根据数字矩阵生成图片
import numpy as np   # 生成随机矩阵

Fig_Path = r"C:\Users\GWT9\Desktop\my.png"   # 保存图片路径

Zodiac = 8  # 取值1-12，将生肖数设置为随机种子，生成R通道矩阵
Constellation = 3  # 取值1-12，将星座数设置为随机种子生成G通道矩阵
Time_Birth = 9  # 取值1-12，将出生时辰设置为随机种子，生成B通道矩阵
Lucky_Number = 7  # 取值1-9，图片的高度转换到179至187之间。1对应187，9对应179，宽度为乘以1.618取整


# 根据数字获取矩阵
def greate_matrix(seed, luck=Lucky_Number, channel='R', size=187):
    """
    根据随机种子，随机获取矩阵，防止出现
    :param seed: 随机种子
    :param luck: 幸运数字
    :param channel: 通道名称
    :param size: 图片的最大高度
    :return: 通道矩阵
    """
    np.random.seed(seed * ord(channel))  # 防止三个通道出现相同的数字
    fig_width = size - luck + 1
    fig_height = int(fig_width * 1.618)
    random_matrix = np.random.randint(0, 256, size=(fig_height, fig_width))
    return random_matrix


R = greate_matrix(Zodiac)  # 获取R通道
G = greate_matrix(Constellation, channel='G')   # 获取G通道
B = greate_matrix(Constellation, channel='B')    # 获取B通道

# 通道矩阵变换维度
R_t = R.reshape(-1, len(R[0]), 1)
G_t = G.reshape(-1, len(G[0]), 1)
B_t = B.reshape(-1, len(B[0]), 1)

#  三个矩阵合成为三维矩阵
my_matrix = np.concatenate([R_t, G_t, B_t], 2)

# 输出图片
R_image = Image.fromarray(my_matrix, mode="RGB")
R_image.show()
R_image.save(Fig_Path)

```

* 结果

 ![专属图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/my.png)!

注意到，上面的图片看不出任何的线条，图案之类，就是由一些不同颜色的像素点构成。这是因为相邻的像素的颜色都是不同的，形不成线条或者任何的图案。下面利用卷积、池化操作，看看效果如何。关于卷积和池化的具体操作点击。





