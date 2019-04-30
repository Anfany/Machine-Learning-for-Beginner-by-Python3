
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


#  根据生肖数+星座数+出生时辰+幸运数字生成专属图片

from PIL import Image  # 根据数字矩阵生成图片
import numpy as np  # 生成随机矩阵

Fig_Path_Original = r"C:\Users\GWT9\Desktop\my_r.png"  # 保存随机图片的路径

Zodiac = 8  # 取值1-12，将生肖数设置为随机种子，生成R通道矩阵
Constellation = 3  # 取值1-12，将星座数设置为随机种子生成G通道矩阵
Time_Birth = 9  # 取值1-12，将出生时辰设置为随机种子，生成B通道矩阵
Lucky_Number = 8  # 取值1-9，图片的宽度为33*幸运数字，高度为宽度除以1.618


# 根据数字获取矩阵
def greate_matrix(seed, luck=Lucky_Number, channel='R', size=33):
    """
    根据随机种子，随机获取矩阵，
    :param seed: 随机种子
    :param luck: 幸运数字
    :param channel: 通道名称，默认为R通道
    :param size: 图片的高度
    :return: 通道矩阵
    """
    np.random.seed(seed * ord(channel))  # 防止三个通道出现相同的数字
    fig_width = size * luck
    fig_height = int(fig_width / 1.618)
    random_matrix = np.random.randint(0, 256, size=(fig_height, fig_width))
    return random_matrix

R = greate_matrix(Zodiac)  # 获取R通道数字矩阵
G = greate_matrix(Constellation, channel='G')  # 获取G通道数字矩阵
B = greate_matrix(Time_Birth, channel='B')  # 获取B通道数字矩阵
# 通道矩阵变换维度
R1 = R.reshape(-1, len(R[0]), 1)
G1 = G.reshape(-1, len(G[0]), 1)
B1 = B.reshape(-1, len(B[0]), 1)
# 三个矩阵合成为三维矩阵
my_matrix_original = np.concatenate([R1, G1, B1], 2)

# 输出图片
R_image = Image.fromarray(np.uint8(my_matrix_original), mode="RGB")
R_image.show()
R_image.save(Fig_Path_Original)

```

* 结果

 ![专属图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/my_r.png)

注意到，上面的图片看不出任何的线条，图案之类，就是由一些不同颜色的像素点构成。这是因为相邻的像素的颜色都是不同的，形不成线条或者任何的图案。下面利用卷积、池化操作，看看效果如何。关于卷积和池化的具体操作点击。

```python
Fig_Path = r"C:\Users\GWT9\Desktop\my.png"  # 保存经过卷积池化后的图片的路径
# 卷积和池化的尺寸
Convolution_Size = [5, 5]  # 注意2个都是奇数，这样才可以找到卷积核的中心。两个最好设置为一样的
Pool_Size = [20, 20]

def Convolution_Pooling(path_matrix, p_number, luck=Lucky_Number, c_size=Convolution_Size, p_size=Pool_Size):
    """
    进行卷积和池化操作，因为一般的池化，会缩小图片的尺寸，在这里需要保证原始尺寸
    此处涉及到的卷积、池化都是概念借鉴，和卷积神经网络中的卷积设置会有差别
    :param path_matrix: 表示图像某个通道的矩阵，此处是二维的
    :param p_number: 随机选取卷积核的种子
    :param luck: 幸运数字，和池化的方法有关系
    :param c_size: 卷积的矩阵大小是固定，但是矩阵中元素的排列和通道的数字有关系，卷积后保持原始尺寸，
    :param p_size: 池化的区域大小是固定的。方法包括2种，最大池化，均值池化，方法和幸运数字的奇偶性有关，奇数最大池化，偶数均值池化。
                        最大池化就是这个区域内的所有数字均变为区域内的最大值。
                        均值池化就是变为这个区域内数字的均值。
    :return: 卷积池化后的矩阵
    """
    # 大于255的设置为255，小于255的设置为0，每个通道的卷积核是不同的

    # 设置该通道矩阵的卷积核
    np.random.seed(p_number)
    c_matrix = -1 + 2 * np.random.random(tuple(c_size))  # 该通道卷积核

    # 开始进行卷积操作，因为要保持图像尺寸，需要在矩阵周围添加0，又称为padding。简单起见，卷积的步长设置为1，因此也就是在
    # 矩阵的上下，各添加c_size[0] // 2行，矩阵的左右，各添加c_size[1] // 2列
    # 获取矩阵的维度
    row, column = path_matrix.shape
    # 列要添加的
    b = np.zeros((row, c_size[0] // 2))
    # 矩阵前后的列添加了
    path_matrix = np.hstack((path_matrix, b))
    path_matrix = np.hstack((b, path_matrix))
    # 行要添加的
    c = np.zeros((c_size[1] // 2, column + 2 * (c_size[0] // 2)))  # 因为列数增加了
    # 矩阵上下的行添加了
    path_matrix = np.vstack((path_matrix, c))
    path_matrix = np.vstack((c, path_matrix))

    # 添加完毕后开始进行卷积操作，和原始尺寸是一样的
    new_matrix = np.zeros((row, column))
    for r in range(row):
        for c in range(column):
            # 计算卷积
            p_matrix = path_matrix[r: r + c_size[0], c: c + c_size[1]]
            # 计算对应元素的乘积
            p = int(np.sum(p_matrix * c_matrix))

            if p < 0:
                new_matrix[r][c] = 0
            elif p > 255:
                new_matrix[r][c] = 255
            else:
                new_matrix[r][c] = p

    # 卷积完毕，下面进行池化，最终的图片的高和宽只保留池化size的最大整数倍
    p_r, p_c = row // p_size[0], column // p_size[1]
    new_pool_matrix = np.zeros((p_r * p_size[0], p_c * p_size[1]))
    if luck % 2 == 1:  # 奇数最大池化
        # 在池化区域内的所有数字变为这些数字的最大值
        for i in range(p_r):
            for j in range(p_c):
                # 区域内的矩阵
                pool_matrix = path_matrix[i * p_size[0]: (i + 1) * p_size[0],
                              j * p_size[1]: (j + 1) * p_size[1]]
                # 获取最大值
                max_num = np.max(pool_matrix)
                # 新的矩阵
                new_pool_matrix[i * p_size[0]: (i + 1) * p_size[0],
                j * p_size[1]: (j + 1) * p_size[1]] = np.tile([max_num], p_size)
    else:
        # 在池化区域内的所有数字变为这些数字的最大值
        for i in range(p_r):
            for j in range(p_c):
                # 区域内的矩阵
                pool_matrix = path_matrix[i * p_size[0]: (i + 1) * p_size[0],
                              j * c_size[1]: (j + 1) * c_size[1]]
                # 获取最大值
                avg_num = int(np.mean(pool_matrix))
                # 新的矩阵
                new_pool_matrix[i * p_size[0]: (i + 1) * p_size[0],
                j * p_size[1]: (j + 1) * p_size[1]] = np.tile([avg_num], p_size)
    return new_pool_matrix

R = greate_matrix(Zodiac)  # 获取R通道数字矩阵
G = greate_matrix(Constellation, channel='G')  # 获取G通道数字矩阵
B = greate_matrix(Time_Birth, channel='B')  # 获取B通道数字矩阵

# 将图像进行卷积和池化
# 卷积池化后的数据
R_t = Convolution_Pooling(R, Zodiac)
G_t = Convolution_Pooling(G, Constellation)
B_t = Convolution_Pooling(B, Time_Birth)

# 通道矩阵变换维度
R_t_1 = R_t.reshape(-1, len(R_t[0]), 1)
G_t_1 = G_t.reshape(-1, len(G_t[0]), 1)
B_t_1 = B_t.reshape(-1, len(B_t[0]), 1)


# #  三个矩阵合成为三维矩阵
my_matrix = np.concatenate([R_t_1, G_t_1, B_t_1], 2)

# 输出图片
R_image = Image.fromarray(np.uint8(my_matrix), mode="RGB")
R_image.show()
R_image.save(Fig_Path)

```

* 结果

![卷积池化后的专属图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/my.png)!

