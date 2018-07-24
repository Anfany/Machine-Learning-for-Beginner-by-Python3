#  # -*- coding：utf-8 -*-
# &Author  AnFany

# 坐标上升(求极大值)/下降(求极小值)法

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False # 显示负号

# 定义目标函数，必须是凸函数，否则只能到达局部最优值
def tar(a, b):
    return 3 * a ** 2 + 5 * b ** 2 + 4 * a * b + 14 * a - 6 * b - 45


# 数据
A = np.arange(-10, 10, 0.2)
B = np.arange(-10, 10, 0.2)
A, B = np.meshgrid(A, B)

# 绘制函数的三维图像
fig = plt.figure()
ax = Axes3D(fig)
sur = ax.plot_surface(A, B, tar(A, B), rstride=10, cstride=10, cmap=cm.YlGnBu_r)
fig.colorbar(sur, shrink=0.3)

# 开始坐标下降法

# 定义初始值
alist = [9]
blist = [9]

funlist = [tar(alist[-1], blist[-1])]

# 定义结束循环的条件

stop = 1

while stop > 0.0000001:
    # a看作已知量，更新b值
    b = (3 - 2 * alist[-1]) / 5
    # b看作已知量，更新a值
    a = - (7 + 2 * blist[-1]) / 3

    alist.append(a)
    blist.append(b)
    funlist.append(tar(a, b))

    stop = abs(funlist[-1] - funlist[-2])

# 结束后，在三维的函数图像中添加路径
ax.plot(alist, blist, funlist, linewidth=4, color='#FFFFFF')
ax.set_xlabel('a值')
ax.set_ylabel('b值')
ax.set_zlabel('目标函数值')
plt.title('坐标下降法求得极小值的路径')

plt.show()


