#-*- coding：utf-8 -*-
# &Author  AnFany
#生成数据
import numpy as np
X_DATA = np.random.randint(12,39,20)
Y_DATA = 2 * X_DATA + np.random.random((1,20))[0] * 10

#引入模型
import linear_regression_AnFany as lr_af   # AnFany
import linear_regression_sklearn as lr_sk  # Sklearn
import TensorFlow_rewrite as lr_tf  # TensorFlow


#绘图对比展示
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文
mpl.rcParams['axes.unicode_minus'] = False   # 负号

#数据格式转换
XDATA= X_DATA.reshape(-1, 1)
YDATA= Y_DATA.reshape(-1, 1)

plt.subplot(2, 2, 1)
plt.plot(X_DATA, Y_DATA, 'o', label='原始数据' )

#绘制AnFany 梯度下降法的结果
resulr_G = lr_af.LinearRegression(0.0002, 90000, 1e-10)#数据不归一的话需要设置较小的学习率
resulr_G.Gradient(XDATA, YDATA)
YDATA_G = resulr_G.predict(XDATA)
plt.plot(XDATA, YDATA_G, '-', label='梯度下降',linewidth=2)


#绘制AnFany 公式法的的结果
resulr_F = lr_af.LinearRegression()
resulr_F.Formula(XDATA, YDATA)
YDATA_F = resulr_F.predict(XDATA)
plt.plot(XDATA, YDATA_F, '^', label='公式法', alpha=10)
plt.legend(loc='upper left')
plt.title('梯度下降VS公式法')




#绘制Sklearn的结果
plt.subplot(2, 2, 2)
plt.plot(X_DATA, Y_DATA, 'o', label='原始数据')
lr_sk.reg.fit(XDATA, YDATA)
YDATA_sk = lr_sk.reg.predict(XDATA)
plt.plot(XDATA, YDATA_sk, '-', label='Sklearn', linewidth=2)
plt.legend(loc='upper left')
plt.title('基于Sklearn')



#绘制TensorFlow的结果
plt.subplot(2, 2, 3)
plt.plot(X_DATA, Y_DATA, 'o', label='原始数据')

YDATA_tf = lr_tf.train_tf(XDATA, YDATA, 0.0002, 90000, 1e-10)
plt.plot(XDATA, YDATA_tf[0], '-', label='TensorFlow', linewidth=2)
plt.legend(loc='upper left')
plt.title('基于TensorFlow')

#计算MSE的函数
def Mse(y1, y2):
    return np.sum((y1 - y2) ** 2) / len(y1)


#输出各自对应的方程
plt.subplot(2, 2, 4)
plt.grid('off')# 关闭网格
plt.text(0.1, 0.7, '梯度下降法：y = %.5f * x + %.5f  MSE = %.7f'%(resulr_G.weights[0], resulr_G.weights[1], Mse(YDATA_G, YDATA)))
plt.text(0.1, 0.6, '公式法：y = %.5f * x + %.5f  MSE = %.7f'%(resulr_F.weights[0], resulr_F.weights[1], Mse(YDATA_F, YDATA)))

plt.text(0.1, 0.4, 'Sklearn：y = %.5f * x + %.5f  MSE = %.7f'%(lr_sk.reg.coef_[0], lr_sk.reg.intercept_[0], Mse(YDATA_sk, YDATA)))

plt.text(0.1, 0.2, 'Tensorflow：y = %.5f * x + %.5f  MSE = %.7f'%(YDATA_tf[1][0], YDATA_tf[2][0], Mse(YDATA_tf[0], YDATA)))
plt.axis('off')# 关闭坐标轴
plt.text(0.5, 0.9, '四种方式的结果对比', fontsize=18)

plt.show()
