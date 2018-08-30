# -*- coding：utf-8 -*-
# &Author AnFany

# 利用Sklearn包实现支持核函数二分类

"""
第一部分：引入库
"""

# 引入心脏病数据
import SVM_Classify_Data as sdata

# 引入库包
from sklearn import svm
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

"""
第二部分：构建函数
"""


# 核函数
def sk_svm_train(intr, labeltr, inte, labelte, kener):
    clf = svm.SVC(kernel=kener)
    # 开始训练
    clf.fit(intr, labeltr)
    #  绘图的标识
    figsign = kener
    #  训练精确度
    acc_train = clf.score(intr, labeltr)
    #  测试精确度
    acc_test = clf.score(inte, labelte)
    #  支持向量的个数
    vec_count = sum(clf.n_support_)
    #  支持向量
    vectors = clf.support_vectors_

    return acc_train, acc_test, vec_count, vectors, figsign


# 结果输出函数
'''
‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
'''
#  K折数据集字典
def result(datadict, he='rbf'):
    sign = []
    trainacc, testacc, vec = [], [], []
    resu = []
    for jj in datadict:
        #  训练数据
        xd = datadict[jj][0][:, :-1]
        yd = datadict[jj][0][:, -1]
        #  测试数据
        texd = datadict[jj][1][:, :-1]
        teyd = datadict[jj][1][:, -1]

        # 开始训练
        resu = sk_svm_train(xd, yd, texd, teyd, he)

        # 储存结果
        trainacc.append(resu[0])
        testacc.append(resu[1])
        vec.append(resu[2])
        sign.append(jj)

    # 绘制多y轴图
    fig, host = plt.subplots()
    # 用来控制多y轴
    par1 = host.twinx()
    #  多条曲线
    p1, = host.plot(sign, trainacc, "b-", marker='8', label='训练', linewidth=2)
    pp, = host.plot(sign, testacc, "b--", marker='*', label='测试', linewidth=2)
    p2, = par1.plot(sign, vec, "r-", marker='8', label='支持向量个数', linewidth=2)
    #  每个轴的内容
    host.set_xlabel("K折数据集")
    host.set_ylabel("分类准确率")
    par1.set_ylabel("个数")
    #  控制每个y轴内容的颜色
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    #  控制每个Y轴刻度数字的颜色以及线粗细
    tkw = dict(size=6, width=3)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)

    #  添加图例
    lines = [p1, pp, p2]
    host.legend(lines, [l.get_label() for l in lines], loc='lower center')

    #  添加标题
    plt.title('K折心脏病数据集SVM分类结果对比 核函数：%s' % resu[-1])

    #  控制每个Y轴刻度线的颜色
    ax = plt.gca()
    ax.spines['left'].set_color('blue')
    ax.spines['right'].set_color('red')

    # 显示图片
    plt.show()


'''第四部分：最终的运行程序'''
if __name__ == "__main__":
    result(sdata.kfold_train_datadict, he='rbf')
