#-*- coding：utf-8 -*-
# &Author  AnFany

import tensorflow as tf
sess = tf.Session()
from Iris_Data import Data as smdata
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
# 将独热编码的类别变为标识为1，2，3的类别
def transign(eydata):
    ysign = []
    for hh in eydata:
        ysign.append([list(hh).index(1) + 1])
    return np.array(ysign)

# 构建函数
def trans_tf(datax, datay, prea, learn_rate=0.005, iter_tiems=40000, error=1e-9):
    # 占位符
    x_data = tf.placeholder(shape=[None, len(datax[0])], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, len(datay[0])], dtype=tf.float32)
    # 逻辑回归参数
    Weight = tf.Variable(tf.random_normal(shape=[len(datax[0]), len(datay[0])]), dtype=tf.float32)
    Bias = tf.Variable(tf.random_normal(shape=[1, len(datay[0])]), dtype=tf.float32)

    model_output = tf.nn.softmax(tf.add(tf.matmul(x_data, Weight), Bias))

    cross_entropy = -tf.reduce_sum(y_target * tf.log(model_output))

    # 正则化
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weight)
    regularizer = tf.contrib.layers.l2_regularizer(scale=2 / 2000)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)


    costfunc = tf.add(cross_entropy, reg_term)

    # 利用不同的方法
    optimizer = tf.train.GradientDescentOptimizer(learn_rate) # 梯度,需要注意学习率不可以设大，如果报错，就需要更改学习率的值
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9) # 动量法，需要注意学习率不可以设大，如果报错，就需要更改学习率的值
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learn_rate, rho=0.55, epsilon=1e-08)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.99, epsilon=1e-08)
    trainstep = optimizer.minimize(costfunc)
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []  # 存储成本函数值
    # 开始训练
    for i in range(iter_tiems):
        sess.run(trainstep, feed_dict={x_data: datax, y_target: datay})
        temp_loss = sess.run(costfunc, feed_dict={x_data: datax, y_target: datay})
        loss_vec.append(temp_loss)
        # 训练提前结束
        if len(loss_vec) > 2:
            if loss_vec[-2] - loss_vec[-1] >= 0 and (loss_vec[-2] - loss_vec[-1]) <= error:
                break

    predata = sess.run(model_output, feed_dict={x_data: prea})
    #转化
    # 选择每一行中最大的数的index
    maxnumber = np.max(predata, axis=1)
    # 预测的类别
    y_pre_type = []
    for jj in range(len(maxnumber)):
        fu = list(predata[jj]).index(maxnumber[jj]) + 1
        y_pre_type.append([fu])


    return loss_vec, np.array(y_pre_type), Weight.eval(session=sess), Bias.eval(session=sess)

tf_result = trans_tf(smdata[0], smdata[1], smdata[0])



print('系数：\n', np.vstack((tf_result[2],tf_result[3])))


print('混淆矩阵：\n', confusion(transign(smdata[1]), tf_result[1]))


# 绘制成本函数图
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体新宋体
mpl.rcParams['axes.unicode_minus'] = False


plt.plot(list(range(len(tf_result[0]))), tf_result[0], '-', linewidth=5)
plt.title('成本函数图')
plt.ylabel('Cost 值')
plt.xlabel('迭代次数')
plt.show()
