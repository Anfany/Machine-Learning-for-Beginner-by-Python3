# -*- coding：utf-8 -*-
# &Author  AnFany

# 可处理多分类， 离散以及连续的变量


from sklearn.tree import DecisionTreeClassifier as skdt  # 引入Sklearn中的决策树模型
import DT_Classify_Data as data  # 引入数据
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
    return len(noshu[noshu == npyuan]) / len(npyuan)


X_data, X_data_test, X_data_pre = LisanToLianxu(X_data, X_data_test, X_data_pre)


#  最终的主函数
if __name__ == '__main__':
    xunliande = []
    yazhengde = []
    yucede = []

    #  针对不同的初始深度，计算正确率
    for i in range(2, 13):
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

    # 绘制图

    # 绘制图
    plt.plot(list(range(2, 13)), xunliande, 'o--', label='训练', lw=2)
    plt.plot(list(range(2, 13)), yazhengde, '*--', label='验证', lw=2)
    plt.plot(list(range(2, 13)), yucede, 's--', label='预测', lw=2)
    plt.xlabel('树的初始深度')
    plt.xlim(1, 14)
    plt.grid()
    plt.ylabel('正确率')
    plt.legend(shadow=True, fancybox=True)
    plt.show()





