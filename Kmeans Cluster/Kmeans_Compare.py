#-*- coding：utf-8 -*-
# &Author  AnFany


# 引入三种方法
import Kmeans_AnFany as K_Af  # 注释165行以后的内容
import Kmeans_Sklearn as K_Sk  # 注释71行以后的内容
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
mpl.rcParams['axes.unicode_minus'] = False
import numpy as np

# 利用sklearn生成数据集
from sklearn.datasets import make_blobs
X, Y = make_blobs(n_samples=600, centers=6, n_features=2)

# 绘制散点图
def fig_scatter(exdata, eydata, titl='训练数据散点图', co=['r', 'g', 'k', 'b', 'y', 'm'], marker=['o','^','H','v','d','>']):
    typeclass = sorted(list(set(eydata)))
    for ii in range(len(typeclass)):
        datax = exdata[eydata == typeclass[ii]]
        plt.scatter(datax[:, 0], datax[:, -1], c=co[ii], s=50, marker=marker[ii])
    plt.title(titl)
    #plt.legend(['%d类'%i for i in typeclass], bbox_to_anchor=(1.2, 0.9))
    plt.xlabel('特征1')
    plt.ylabel('特征2')

# 调用不同的方法

# AnFany
kresult = K_Af.op_kmeans(X, countcen=6)


# Sklearn
sk = K_Sk.KMeans(init='k-means++', n_clusters=6, n_init=10)

train = sk.fit(X)
result = sk.predict(X)
skru = K_Sk.trans(result)



#绘制算法后的类别的散点图
def sca(Xdata, Center, signdict, co=['r', 'g', 'y', 'b', 'c', 'm'], marker=['o','^','H','s','d','*'], titl = 'AnFany 结果'):
    du = 1
    for jj in signdict:
        xdata = Xdata[signdict[jj]]
        plt.scatter(xdata[:, 0], xdata[:, -1], c=co[jj], s=50, marker=marker[jj], label='%d类' % jj)  # 绘制样本散点图
    for ss in Center:
        if du:
            plt.scatter(ss[0], ss[1], c='k', s=100, marker='8', label='类别中心') #绘制类别中心点
            du = 0
        else:
            plt.scatter(ss[0], ss[1], c='k', s=100, marker='8')  # 绘制类别中心点

    plt.legend(bbox_to_anchor=(1.2, 1))
    plt.title(titl)
    plt.xlabel('特征1')
    plt.ylabel('特征2')

# 定义欧几里得距离
def dis(sample, center):
    cen = np.array([center])
    sample = np.array(sample)
    if len(sample) != 0:
        usb = np.sum((sample - cen) ** 2, axis=1) ** 0.5
        return usb
    else:
        return 0
# 计算最终的分类结果的成本值
def Cost(Xdata, typedict):
    center = {}
    for kk in typedict:
        center[kk] = np.mean(Xdata[typedict[kk]], axis=0) # 均值
    cio = 0
    for cc in typedict:
        cio += np.sum(dis(Xdata[typedict[cc]], center[cc]))
    return cio

# 最终的结果展示
plt.subplot(2, 2, 1)
fig_scatter(X, Y)

plt.subplot(2, 2, 2)
sca(X, kresult[0], kresult[2])

plt.subplot(2, 2, 3)
sca(X, train.cluster_centers_, skru, titl='Sklearn 结果')

plt.subplot(2, 2, 4)
plt.axis('off')
plt.text(0.3, 0.6, 'AnFany 最终的分类成本值为：%.5f'%Cost(X, kresult[2]))
plt.text(0.3, 0.3, 'Sklearn 最终的分类成本值为：%.5f'%Cost(X, skru))

plt.show()










