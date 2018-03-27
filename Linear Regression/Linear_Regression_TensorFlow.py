#-*- coding：utf-8 -*-
# &Author  AnFany

#获得数据
from Boston_Data import model_data as lrdata

# 训练数据x
x_train_data = lrdata[0]
#预测数据x
x_predict_data = lrdata[2]

# y数据(样本数，1)
y_train_data = lrdata[1]
y_predict_data = lrdata[3]


# 引入库
import tensorflow as tf
import numpy as np

# 参数
learn_rate = 0.2
iter_times = 1000
error = 1e-9

# 预先输入的数据
x_data = tf.placeholder(shape=[None, len(x_train_data[0])], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 线性回归参数
Weight = tf.Variable(tf.random_normal(shape=[len(x_train_data[0]), 1]))
Bias = tf.Variable(tf.random_normal(shape=[1, 1]))
y_out = tf.add(tf.matmul(x_data, Weight), Bias)

#L2正则化
tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weight)
regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 6000)
reg_term = tf.contrib.layers.apply_regularization(regularizer)


# 损失函数
cost = tf.reduce_mean(tf.square(y_out - y_data)) + reg_term

# 初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)


#误差存储
costfunc = []

for i in range(iter_times):

    sess.run(optimizer, feed_dict={x_data: x_train_data, y_data: y_train_data})

    y_step_out = sess.run(y_out, feed_dict={x_data: x_train_data})

    loss = sess.run(cost, feed_dict={y_out: y_step_out, y_data: y_train_data})

    costfunc.append(loss)

    # 提前结束循环的机制
    if len(costfunc) > 1:
        if 0 < costfunc[-2] - costfunc[-1] < error:
            break



# 用于预测数据的预测值
predict_result = sess.run(y_out, feed_dict={x_data: x_predict_data})
# 用于训练数据的预测值
train_pre_result = sess.run(y_out, feed_dict={x_data: x_train_data})



import matplotlib.pyplot as plt#绘图
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False   #负号

# 绘图函数
def figure(title, *datalist):
    for jj in datalist:
        plt.plot(jj[0], '-', label=jj[1], linewidth=2)
        plt.plot(jj[0], 'o')
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.show()

#计算R2的函数
def getR(ydata_tr, ydata_pre):
    sum_error = np.sum(((ydata_tr - np.mean(ydata_tr)) ** 2))
    inexplicable = np.sum(((ydata_tr - ydata_pre) ** 2))
    return 1 - inexplicable / sum_error



#绘制误差图
figure('误差图 最终的MSE = %.4f'%(costfunc[-1]), [costfunc, 'error'])


#绘制预测值与真实值图
figure('预测值与真实值图 模型的' + r'$R^2=%.4f$'%(getR(y_train_data, train_pre_result)), [predict_result, '预测值'],[y_predict_data,'真实值'])
plt.show()

#线性回归的参数
print('线性回归的系数为:\n w = %s, \nb= %s'%(Weight.eval(session=sess), Bias.eval(session=sess)))
