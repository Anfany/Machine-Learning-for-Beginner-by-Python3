# -*- coding：utf-8 -*-
# &Author  AnFany

from Heart_Data import model_data  as H_Data
import numpy as np


#计算混淆矩阵
from prettytable import PrettyTable
def confusion(realy, outy):
    mix = PrettyTable()
    type = sorted(list(set(realy.T[0])), reverse=True)
    mix.field_names = [' '] + ['预测:%d类'%si for si in type]
    # 字典形式存储混淆矩阵数据
    cmdict = {}
    for jkj in type:
        cmdict[jkj] = []
        for hh in type:
            hu = len(['0' for jj in range(len(realy)) if realy[jj][0] == jkj and outy[jj][0] == hh])
            cmdict[jkj].append(hu)
    # 输出表格
    for fu in type:
        mix.add_row(['真实:%d类'%fu] + cmdict[fu])
    return mix



class LRReg:
    def __init__(self, learn_rate=0.5, iter_times=40000, error=1e-9, cpn='L2'):
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.error = error
        self.cpn = cpn

    # w和b合为一个参数，也就是x最后加上一列全为1的数据。
    def trans(self, xdata):
        one1 = np.ones(len(xdata))
        xta = np.append(xdata, one1.reshape(-1, 1), axis=1)
        return xta

    # 梯度下降法
    def Gradient(self, xdata, ydata, func=trans):
        xdata = func(self, xdata)
        # 系数w,b的初始化
        self.weights = np.zeros((len(xdata[0]), 1))
        # 存储成本函数的值
        cost_function = []

        for i in range(self.iter_times):
            # 得到回归的值
            y_predict = np.dot(xdata, self.weights)

            # Sigmoid函数的值
            s_y_pre = 1/ (1 + np.exp(-y_predict))

            # 计算最大似然取对数后的值
            like = np.dot(ydata.T, np.log(s_y_pre)) + np.dot((1 - ydata).T, np.log(1- s_y_pre))

            # 正则化
            if self.cpn == 'L2':
                # 成本函数中添加系数的L2范数
                l2norm = 0.5 * np.dot(self.weights.T, self.weights) / len(xdata)
                cost = -like / len(xdata) + l2norm

                grad_W = np.dot(xdata.T, (s_y_pre - ydata)) / len(xdata) + 0.9 * self.weights / len(xdata)

            else:
                cost = -like / (len(xdata))
                grad_W = np.dot(xdata.T, (s_y_pre - ydata)) / len(xdata)

            cost_function.append(cost[0][0])

            # 训练提前结束
            if len(cost_function) > 2:
                if 0 <= cost_function[-1] - cost_function[-2] <= self.error:
                    break

            #更新
            self.weights = self.weights - self.learn_rate * grad_W

        return self.weights, cost_function

    # 预测
    def predict(self, xdata, func=trans):
        pnum = np.dot(func(self, xdata), self.weights)
        s_pnum = 1/ (1 + np.exp(-pnum))
        latnum = [[1] if jj[0] >= 0.5 else [0] for jj in s_pnum]
        return latnum



lr_re = LRReg()

lf = lr_re.Gradient(H_Data[0], H_Data[1])

print('系数为：\n', lr_re.weights)

fdatd = lr_re.predict(H_Data[0])

print('混淆矩阵：\n', confusion(H_Data[1], fdatd))

# 绘制成本函数图
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体

plt.plot(list(range(len(lf[1]))), lf[1], '-', linewidth=5)
plt.title('成本函数图')
plt.ylabel('Cost 值')
plt.xlabel('迭代次数')
plt.show()
