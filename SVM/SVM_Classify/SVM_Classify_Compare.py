# -*- coding：utf-8 -*-
# &Author AnFany

#  AnFany与Sklearn的结果对比

"""
第一部分：生成数据
"""
import numpy as np

data = np.random.rand(900, 2)


#  将数据分类, 样本数定为200
def fenlei(exdata, samples=200):
    feadata, ladata = [], []
    for jj in exdata:
        if jj[1] - jj[0] - 0.55 > 0:
            feadata.append(jj)
            ladata.append(1)
        elif jj[1] - jj[0] - 0.45 < 0 and jj[1] - jj[0] - 0.05 > 0:
            feadata.append(jj)
            ladata.append(-1)
        elif jj[1] - jj[0] + 0.05 < 0 and jj[1] - jj[0] + 0.45 > 0:
            feadata.append(jj)
            ladata.append(1)
        elif jj[1] - jj[0] + 0.55 < 0:
            feadata.append(jj)
            ladata.append(-1)
        else:
            pass
        if len(feadata) >= samples:
            break
    return np.array(feadata), np.array(ladata)


traindata = fenlei(data)

"""
第二部分：SVM训练，引入模型
"""
#  AnFany
import AnFany_SVM_Classify as An_svm
#  Sklearn
from sklearn import svm


# 得到绘制决策边界的数据
def bound(featuredata, labeldata, mol, ke='rbf'):
    # 获得属性数据2个维度的最大、最小值
    xmin, xmax = min(featuredata[:, :-1]) - 0.1, max(featuredata[:, :-1]) + 0.1
    ymin, ymax = min(featuredata[:, :-1]) - 0.1, max(featuredata[:, :-1]) + 0.1

    # 生成网格
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.02),
                         np.arange(ymin, ymax, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if mol == 'AnFany':

        # 开始引入模型
        model = An_svm.SVM(feature=featuredata, labels=labeldata, kernel=ke, times=900)
        #  开始训练
        An_svm.platt_smo(model)
        # 返回支持向量
        support = model.feature[model.alphas > 0]
        # 开始预测
        prepre = An_svm.predict(model, grid_points)
    else:
        # 开始引入模型
        clf = svm.SVC(kernel=ke, max_iter=900, gamma=8)
        #  开始训练
        clf.fit(featuredata, labeldata)
        # 返回支持向量
        support = clf.support_vectors_
        # 开始预测
        prepre = clf.predict(grid_points)
    return xx, yy, prepre, support


# 绘图库
import matplotlib.pyplot as plt
from pylab import mpl
from matplotlib import cm

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号



#  绘制图
def figdata(featuredata, labeldata, osel='Sklearn'):
    #  首先是将数据分为2类
    neg = featuredata[labeldata == 1]
    pos = featuredata[labeldata == -1]

    # 绘图
    fig, ax = plt.subplots()


    # 绘制决策边界
    xx, yy, opre, suo = bound(featuredata, labeldata, mol=osel)

    opr = opre.reshape(xx.shape)

    ax.contourf(xx, yy, opr, alpha=0.8, cmap=cm.PuBu_r)

    # 绘制原始数据图
    # 一类
    ax.scatter(neg[:, :-1], neg[:, -1], c='r', alpha=0.9, marker='8', s=50)
    # 负一类
    ax.scatter(pos[:, :-1], pos[:, -1], c='b', alpha=0.9, marker='8', s=50)

    # 绘制支持向量
    ax.scatter(suo[:, :-1], suo[:, -1], s=130, c='', edgecolors='k', alpha=0.9, label='支持向量(%s个)' % len(suo), marker='s')

    # 图例
    plt.legend()

    # 图的设置
    plt.title('支持向量机分类结果展示：方法%s' % osel)
    plt.xlabel('第一维')
    plt.ylabel('第二维')

    # 显示图
    plt.show()


'''第四部分：最终的运行程序'''
if __name__ == "__main__":
    figdata(traindata[0], traindata[1], osel='AnFany')
    figdata(traindata[0], traindata[1])
