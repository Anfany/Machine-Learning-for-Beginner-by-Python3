# -*- coding：utf-8 -*-
# &Author  AnFany

# 回归树， 可处理离散以及连续的变量


from sklearn.tree import DecisionTreeRegressor as skdt  # 引入Sklearn中的决策树模型
import Data_DT_Regression as data  # 引入数据
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # 绘制图
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


# 因为Sklearn只能处理数值型的数据，因此需要爸X数据中值为字符串的变量转变为数值型。
# 这里未采用独热化编码，一是考虑数据量会递增 二是两者的效果应该相差不大

# 训练数据X
X_data = data.dt_data[0]['train'][:, :-1]
# Y
Y_data = data.dt_data[0]['train'][:, -1:].T[0]

# 验证数据X
X_data_test = data.dt_data[0]['test'][:, :-1]
# Y
Y_data_test = data.dt_data[0]['test'][:, -1:].T[0]


# 预测数据
X_data_pre = data.test_data[:, :-1]
# Y
Y_data_pre = data.test_data[:, -1:].T[0]


# 定义更改所有离散变量的函数
def LisanToLianxu(tdata, ydata, pdata):
    for jj in range(len(tdata[0])):
        try:
            tdata[0][jj] + 2
            pass
        except TypeError:
            # 这是离散的需要处理
            #  默认 训练数据中包括所有的本变量的值
            zhiset = list(set(list(tdata[:, jj:(jj + 1)].T[0])))

            # 为每个值赋予一个数

            numdict = {i: zhiset.index(i) for i in zhiset}

            # 开始为每一个数据中的这一列赋值
            def zhihuan(tdata, jj, exdict=numdict):
                # 这里需要建立一个DataFrame
                du = pd.DataFrame()

                du['zhi'] = tdata[:, jj:(jj + 1)].T[0]

                du['zhi'] = du['zhi'].map(exdict)

                tdata[:, jj:(jj + 1)] = np.array([du['zhi'].values]).T
                return tdata

            tdata = zhihuan(tdata, jj)
            ydata = zhihuan(ydata, jj)
            pdata = zhihuan(pdata, jj)

    return tdata, ydata, pdata


# 定义计算正确率的函数

def CorrectRate(yuanshileibei, shuchuleibie):
    npyuan = np.array(yuanshileibei)
    noshu = np.array(shuchuleibie)
    cha = npyuan - noshu
    return np.sum([i ** 2 for i in cha]) / len(npyuan)


X_data, X_data_test, X_data_pre = LisanToLianxu(X_data, X_data_test, X_data_pre)



#  数据量较大，折线图对比不容易看清，训练数据随机选取200条，验证和预测随机选取100条展示
def selet(prdata, reda, count=200):
    if len(reda) <= count:
        return prdata, reda
    fu = np.arange(len(reda))

    du = np.random.choice(fu, count)

    return np.array(prdata)[du], np.array(reda)[du]


#  最终的主函数
if __name__ == '__main__':
    xunliande = []
    yazhengde = []
    yucede = []

    #  针对不同的初始深度，计算正确率
    for i in range(2, 25):
        clf = skdt(max_depth=i).fit(X_data, Y_data)
        # 训练
        Y_data_shu = clf.predict(X_data)

        xunliande.append(CorrectRate(Y_data, Y_data_shu))

        # 验证
        Y_data_test_shu = clf.predict(X_data_test)

        yazhengde.append(CorrectRate(Y_data_test, Y_data_test_shu))

        # 预测
        Y_data_pre_shu = clf.predict(X_data_pre)

        yucede.append(CorrectRate(Y_data_pre, Y_data_pre_shu))

    # 选择最优的深度

    zonghe = [j + k for j, k in zip(yazhengde, yucede)]
    # 选择最小的值,
    zuiyoushendu = zonghe.index(min(zonghe)) + 2

    # 绘制图
    plt.plot(list(range(2, 25)), xunliande, 'o--', label='训练', lw=2)
    plt.plot(list(range(2, 25)), yazhengde, '*--', label='验证', lw=2)
    plt.plot(list(range(2, 25)), yucede, 's--', label='预测', lw=2)
    plt.xlabel('树的初始深度')
    plt.xlim(1, 25)
    plt.grid()
    plt.ylabel('MSE')
    plt.legend(shadow=True, fancybox=True)
    plt.title('树的最佳深度：%d' % zuiyoushendu)
    plt.show()

    # 绘制真实值和预测值得对比曲线

    clf = skdt(max_depth=zuiyoushendu).fit(X_data, Y_data)
    # 训练
    Y_data_shu = clf.predict(X_data)
    # 验证
    Y_data_test_shu = clf.predict(X_data_test)
    # 预测
    Y_data_pre_shu = clf.predict(X_data_pre)

    # 绘制真实值和预测值得曲线
    plt.subplot(211)
    a, b = selet(Y_data_shu, Y_data)
    plt.plot(list(range(len(a))), a, 'o--', list(range(len(b))), b, '8-')

    plt.legend(['预测', '真实'])
    plt.title('训练数据')


    plt.subplot(223)
    c, d = selet(Y_data_test_shu, Y_data_test, count=100)
    plt.plot(list(range(len(c))), c, 'o--', list(range(len(d))), d, '8-')
    plt.legend(['预测', '真实'])
    plt.title('验证数据')

    plt.subplot(224)
    e, f = selet(Y_data_pre_shu, Y_data_pre, count=100)
    plt.plot(list(range(len(e))), e, 'o--', list(range(len(f))), f, '8-')
    plt.legend(['预测', '真实'])
    plt.title('预测数据')

    plt.show()







