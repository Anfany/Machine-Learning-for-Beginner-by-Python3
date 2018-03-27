#-*- coding：utf-8 -*-
# &Author  AnFany

#获得数据
import Boston_Data as lrdata
import numpy as np

#创建线性回归的类
class LinearRegression:

    def __init__(self, learn_rate=0.2, iter_times=200000, error=1e-9):
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.error = error

    #w和b合为一个参数，也就是x最后加上一列全为1的数据。
    def Trans(xdata):
        one1 = np.ones(len(xdata))
        xta = np.append(xdata, one1.reshape(-1, 1), axis=1)
        return xta

    #梯度下降法
    def Gradient(self, xdata, ydata, func=Trans):
        xdata = func(xdata)
        #系数w,b的初始化
        self.weights = np.zeros((len(xdata[0]), 1))
        #存储成本函数的值
        cost_function = []

        for i in range(self.iter_times):
            #得到回归的值
            y_predict = np.dot(xdata, self.weights)

            # 最小二乘法计算误差
            cost = np.sum((y_predict - ydata) ** 2) / len(xdata)
            cost_function.append(cost)

            #计算梯度
            dJ_dw = 2 * np.dot(xdata.T, (y_predict - ydata)) / len(xdata)


            #更新系数w,b的值
            self.weights = self.weights - self.learn_rate * dJ_dw


            #提前结束循环的机制
            if len(cost_function) > 1:
                if cost_function[-1] < cost_function[-2] and cost_function[-2] - cost_function[-1] < self.error:
                    break

        return self.weights, cost_function


    #根据公式
    def Formula(self, xdata, ydata, func=Trans):
        xdata = func(xdata)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(xdata.T, xdata)), xdata.T), ydata)
        return self.weights

    #预测
    def predict(self, xdata, func=Trans):
        return np.dot(func(xdata), self.weights)

#绘图
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体

def figure(title, *datalist):
    for jj in datalist:
        plt.plot(jj[0], '-', label=jj[1], linewidth=4)
    plt.title(title)
    plt.legend()
    plt.show()

#计算R2的函数
def getR(ydata_tr, ydata_pre):
    sum_error = np.sum(((ydata_tr - np.mean(ydata_tr)) ** 2))
    inexplicable = np.sum(((ydata_tr - ydata_pre) ** 2))
    return 1 - inexplicable / sum_error


regressor = LinearRegression()
#开始训练
train_error = regressor.Gradient(lrdata.model_data[0], lrdata.model_data[1])

#用于预测数据的预测值
predict_result = regressor.predict(lrdata.model_data[2])
#用于训练数据的预测值
train_pre_result = regressor.predict(lrdata.model_data[0])



#绘制误差图
figure('误差图', [train_error[1], 'error'])

#绘制预测值与真实值图
figure('预测值与真实值图 模型的' + r'$R^2=%.4f$'%(getR(lrdata.model_data[1], train_pre_result)), [predict_result, '预测值'],[lrdata.model_data[3],'真实值'])
plt.show()

#线性回归的参数
print(train_error[0])
