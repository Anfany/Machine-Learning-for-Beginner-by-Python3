#-*- coding：utf-8 -*-
# &Author  AnFany

#获得数据
from Boston_Data import model_data as lrdata

#引入包
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


#训练数据x
x_train_data = lrdata[0]
#预测数据x
x_predict_data = lrdata[2]


#y数据shape转换(样本数，1)==> (样本数，)
y_train_data = lrdata[1].T[0]
y_predict_data = lrdata[3].T[0]


reg = linear_model.LinearRegression()

#开始训练
reg.fit (x_train_data, y_train_data)

#预测数据的预测值
predict_result = reg.predict(x_predict_data)

#训练数据的预测值
train_pre_result = reg.predict(x_train_data)


import matplotlib.pyplot as plt # 绘图
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
mpl.rcParams['axes.unicode_minus'] = False # 负号
# 绘图函数
def figure(title, *datalist):
    for jj in datalist:
        plt.plot(jj[0], '-', label=jj[1], linewidth=2)
        plt.plot(jj[0], 'o')
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.show()


train_error = [mean_squared_error(y_train_data, [np.mean(y_train_data)] * len(y_train_data)), mean_squared_error(y_train_data, train_pre_result)]

#绘制误差图
figure('误差图 最终的MSE = %.4f'%(train_error[-1]), [train_error, 'error'])

#绘制预测值与真实值图
figure('预测值与真实值图 模型的' + r'$R^2=%.4f$'%(r2_score(train_pre_result, y_train_data)), [predict_result, '预测值'],[y_predict_data,'真实值'])
plt.show()

#线性回归的参数
print('线性回归的系数为:\n w = %s'%(reg.coef_))





