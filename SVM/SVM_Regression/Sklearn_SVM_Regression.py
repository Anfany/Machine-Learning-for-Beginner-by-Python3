# -*- coding：utf-8 -*-
# &Author AnFany

# 利用Sklearn包实现支持核函数回归

"""
第一部分：引入库
"""

# 引入部分的北京PM2.5数据
import SVM_Regression_Data as rdata

# 引入库包
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

"""
第二部分：构建函数
"""


# 核函数
def sk_svm_train(intr, labeltr, inte, kener):
    clf = svm.SVR(kernel=kener)
    #  开始训练
    clf.fit(intr, labeltr)
    #  训练输出
    tr = clf.predict(intr)
    #  预测输出
    pr = clf.predict(inte)

    return tr, pr


# 结果输出函数
'''
‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
'''
#  数据集
def result(data, he='rbf'):
    # 训练、预测的网络输出
    trainacc, testacc = [], []

    xd = data[0]
    yd = data[1].T[0]
    #  测试数据
    texd = data[2]
    teyd = data[3].T[0]

    # 开始训练
    resu = sk_svm_train(xd, yd, texd, he)

    tra = resu[0] * (data[4][1] - data[4][0]) + data[4][0]

    pre = resu[1] * (data[4][1] - data[4][0]) + data[4][0]

    ydd = data[1].T[0] * (data[4][1] - data[4][0]) + data[4][0]

    teyd = data[3].T[0] * (data[4][1] - data[4][0]) + data[4][0]

    return ydd, tra, teyd, pre


# 绘图的函数
def huitu(suout, shiout, c=['b', 'k'], sign='训练', cudu=3):

    # 绘制原始数据和预测数据的对比
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(suout))), suout, c=c[0], linewidth=cudu, label='%s：算法输出' % sign)
    plt.plot(list(range(len(shiout))), shiout, c=c[1], linewidth=cudu, label='%s：实际值' % sign)
    plt.legend(loc='best')
    plt.title('原始数据和向量机输出数据的对比')

    # 绘制误差和0的对比图
    plt.subplot(2, 2, 3)
    plt.plot(list(range(len(suout))), suout - shiout, c='r', linewidth=cudu, label='%s：误差' % sign)
    plt.plot(list(range(len(suout))), list(np.zeros(len(suout))), c='k', linewidth=cudu, label='0值')
    plt.legend(loc='best')
    plt.title('误差和0的对比')
    # 需要添加一个误差的分布图
    plt.subplot(2, 2, 4)
    plt.hist(suout - shiout, 50, facecolor='g', alpha=0.75)
    plt.title('误差直方图')
    # 显示
    plt.show()


'''第四部分：最终的运行程序'''
if __name__ == "__main__":
    datasvr = rdata.model_data
    realtr, outtri, realpre, poupre = result(datasvr, he='rbf')

    huitu(realtr, outtri, c=['b', 'k'], sign='训练', cudu=1.5)

    huitu(realpre, poupre, c=['b', 'k'], sign='预测', cudu=1.5)



