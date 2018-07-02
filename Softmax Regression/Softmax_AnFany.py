#-*- coding：utf-8 -*-
# &Author  AnFany

from Iris_Data import Data as smdata
import numpy as np


class LRReg:
    def __init__(self, learn_rate=0.9, iter_times=40000, error=1e-17):
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.error = error

    # w和b合为一个参数，也就是x最后加上一列全为1的数据。
    def trans(self, xdata):
        one1 = np.ones(len(xdata))
        xta = np.append(xdata, one1.reshape(-1, 1), axis=1)
        return xta

    # 梯度下降法
    def Gradient(self, xdata, ydata, func=trans):
        xdata = func(self, xdata)
        # 系数w,b的初始化
        self.weights = np.zeros((len(xdata[0]), len(ydata[0])))
        # 存储成本函数的值
        cost_function = []

        for i in range(self.iter_times):
            # 计算np.exp(X.W)的值
            exp_xw = np.exp(np.dot(xdata, self.weights))

            #计算y_predict每一行的和值
            sumrow = np.sum(exp_xw, axis=1).reshape(-1, 1)

            # 计算除去和值得值
            devi_sum = exp_xw / sumrow

            # 计算减法
            sub_y = ydata - devi_sum

            # 得到梯度
            grad_W = -1 / len(xdata) * np.dot(xdata.T, sub_y)


            # 正则化
            # 成本函数中添加系数的L2范数
            l2norm = np.sum(0.5 * np.dot(self.weights.T, self.weights) / len(xdata))

            last_grad_W = grad_W + 0.002 * self.weights / len(xdata)

            # 计算最大似然的对数的值
            likehood = np.sum(ydata * devi_sum)

            cost = - likehood / len(xdata) + l2norm

            cost_function.append(cost)

            # 训练提前结束
            if len(cost_function) > 2:
                if 0 <= cost_function[-2] - cost_function[-1] <= self.error:
                    break

            #更新
            self.weights = self.weights - self.learn_rate * last_grad_W

        return self.weights, cost_function

    # 预测
    def predict(self, xdata, func=trans):
        pnum = np.dot(func(self, xdata), self.weights)
        # 选择每一行中最大的数的index
        maxnumber = np.max(pnum, axis=1)
        # 预测的类别
        y_pre_type =[]
        for jj in range(len(maxnumber)):
            fu = list(pnum[jj]).index(maxnumber[jj]) + 1
            y_pre_type.append([fu])
        return np.array(y_pre_type)

# 将独热编码的类别变为标识为1，2，3的类别
def transign(eydata):
    ysign = []
    for hh in eydata:
        ysign.append([list(hh).index(1) + 1])
    return np.array(ysign)

#计算混淆矩阵
from prettytable import PrettyTable
def confusion(realy, outy, method='AnFany'):
    mix = PrettyTable()
    type = sorted(list(set(realy.T[0])), reverse=True)
    mix.field_names = [method] + ['预测:%d类'%si for si in type]
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

# 主函数
if __name__ == '__main__':
    lr_re = LRReg()
    lf = lr_re.Gradient(smdata[0], smdata[1])

    y_calss_pre = lr_re.predict(smdata[0])
    print('系数：\n', lr_re.weights)

    print('混淆矩阵：\n', confusion(transign(smdata[1]), y_calss_pre))

    # 绘制成本函数图
    import matplotlib.pyplot as plt
    from pylab import mpl  # 作图显示中文

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
    mpl.rcParams['axes.unicode_minus'] = False

    plt.plot(list(range(len(lf[1]))), lf[1], '-', linewidth=5)
    plt.title('成本函数图')
    plt.ylabel('Cost 值')
    plt.xlabel('迭代次数')
    plt.show()
    
