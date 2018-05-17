# -*- coding：utf-8 -*-
# &Author  AnFany

# 利用神经网络拟合函数

# 生成训练数据
import numpy as np
import BPNN_DATA_Reg as bp
#  输入数据
xdata = np.array(np.linspace(-80, 80, 500)).reshape(-1, 1)

#  输出数据
y1data = (np.sin(xdata) + np.cos(xdata)) / 2
y2data = (np.sin(xdata) - np.cos(xdata)) * xdata

#  两个输出数据和二为一，
ydata = np.hstack((y1data, y2data))

# 数据标准化
xdata = (xdata - np.mean(xdata, axis=0)) / np.std(xdata, axis=0)

# 最大与最小值
maxnum = np.max(ydata, axis=0, keepdims=True)

minnum = np.min(ydata, axis=0, keepdims=True)

norm_ydata = (ydata - minnum) / (maxnum - minnum)

# 分为训练数据和测试数据
model_data = bp.divided(xdata, norm_ydata, percent=0.1)

# 训练的输入、输出
train_x_in = model_data[0]
train_y_out = model_data[1]

# 预测的输入、输出
pre_x_in = model_data[2]
pre_y_out = model_data[3]


# 引入AnFany以及TensorFlow方法

import AnFany_BPNN_Regression as An_Bpnn  
import TensorFlow_BPNN_Regression as Ten_Bpnn  


# AnFany方法
# # 开始训练数据
bpnn = An_Bpnn.BPNN(train_x_in, train_y_out, learn_rate=0.002, son_samples=50, iter_times=100000000, \
                    hidden_layer=[190, 190, 190], break_error=0.005)
bpnn_train = bpnn.train_adam()

# 训练结果展示
train_output = An_Bpnn.trans(bpnn.predict(train_x_in), minnum, maxnum)
An_Bpnn.figure(An_Bpnn.trans(train_y_out, minnum, maxnum), train_output, le='训练', width=4)

pre_output = An_Bpnn.trans(bpnn.predict(pre_x_in), minnum, maxnum)
An_Bpnn.figure(An_Bpnn.trans(pre_y_out, minnum, maxnum), pre_output, le='预测', width=2)

An_Bpnn.costfig(bpnn_train[2])


# TensorFlow方法
# 开始训练数据
tfrelu = Ten_Bpnn.Ten_train(train_x_in, train_y_out, pre_x_in, break_error=0.005, learn_rate=0.003,\
                            itertimes=100000000000, hiddennodes=120)
train_output = Ten_Bpnn.trans(tfrelu[0], minnum, maxnum)

An_Bpnn.figure(Ten_Bpnn.trans(train_y_out, minnum, maxnum), train_output, le='训练', width=4)

An_Bpnn.figure(Ten_Bpnn.trans(pre_y_out, minnum, maxnum), Ten_Bpnn.trans(tfrelu[1], minnum, maxnum), le='预测', width=4)


An_Bpnn.costfig(tfrelu[2])


