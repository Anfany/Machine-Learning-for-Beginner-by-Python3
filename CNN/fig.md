
### 图像的结构

图像是由一个个像素点构成，每个像素点有三个通道，分别对应R，G，B。彩色RGB图像其实是一个三维矩阵，矩阵中的每个数字(0到255)代表的是一个像素一个通道的灰度。下面举例说明：

![彩色图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af.png)![图片描述](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/af_doc.png)![像素图片](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/CNN/afpixel.png)

下面读取这个图片的三维矩阵：
```python
# -*- coding：utf-8 -*-
# &Author  AnFany

from skimage import io

fig_path = r"C:\Users\GWT9\Desktop\af.png"
matrix = io.imread(fig_path)  
print(matrix, matrix.shape)
```

* 结果
```
[[[137 198 182]
  [126 189 172]
  [112 178 164]
  ...
  [106 220 210]
  [ 94 208 198]
  [ 96 210 202]]

 [[153 214 198]
  [171 233 218]
  [154 220 206]
  ...
  [114 226 214]
  [100 212 200]
  [116 227 218]]

 [[144 206 191]
  [137 199 184]
  [137 201 187]
  ...
  [112 220 207]
  [106 214 201]
  [125 235 224]]

 ...

 [[218 232 219]
  [216 230 217]
  [206 222 211]
  ...
  [136 202 188]
  [140 204 190]
  [140 203 192]]

 [[221 235 222]
  [210 224 211]
  [211 227 216]
  ...
  [139 203 189]
  [143 205 192]
  [145 207 194]]

 [[210 227 211]
  [199 216 200]
  [213 227 214]
  ...
  [142 204 193]
  [147 207 197]
  [151 211 199]]] (189, 149, 3)
```

可以得出图片的数字矩阵维度为(189, 149, 3)。第一个像素(最左上角)对应的RGB就是(137 198 182)。


