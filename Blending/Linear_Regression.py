#-*- coding：utf-8 -*-
# &Author  AnFany

# 线性回归模型
import numpy as np


#创建线性回归的类
class LinearRegression:
    #w和b合为一个参数，也就是x最后加上一列全为1的数据

    def __init__(self, learn_rate=0.0000002, iter_times=200000, error=1e-9):
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.error = error

    def Trans(self, xdata):
        one1 = np.ones(len(xdata))
        xta = np.append(xdata, one1.reshape(-1, 1), axis=1)
        return xta

    #梯度下降法
    def Gradient(self, xdata, ydata):
        xdata = self.Trans(xdata)
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
                if 0 < cost_function[-2] - cost_function[-1] < self.error:
                    break

        return self.weights, cost_function


    #根据公式
    def Formula(self, xdata, ydata):
        xdata = self.Trans(xdata)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(xdata.T, xdata)), xdata.T), ydata)
        y_predict = np.dot(xdata, self.weights)
        cost = [np.sum((ydata - np.mean(ydata)) ** 2) / len(xdata)]  # 开始是以y值得平均值作为预测值计算cost
        cost += [np.sum((y_predict - ydata) ** 2) / len(xdata)]  # 利用公式，一次计算便得到参数的值，不需要迭代。
        return self.weights, cost  # 包括2个值

    #预测
    def predict(self, xdata):
        return np.dot(self.Trans(xdata), self.weights)



