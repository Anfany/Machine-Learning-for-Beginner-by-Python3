# -*- coding：utf-8 -*-
# &Author AnFany

# 基于TensorFlow实现支持向量机二分类
# 防止精确率的不稳定性，不采用batchsize的方式训练，每一次训练都是全部样本
# 因此不适用于样本数据量较大的情况

"""
第一部分：引入库
"""

# 引入库包
import numpy as np

# 引入部分的北京PM2.5数据
import SVM_Regression_Data as rdata

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 绘图
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

"""
第二部分：构造函数
"""


# keneral： 线性可分lin1   核函数： lin：线性核函数   poly：多项式核函数   rbf：高斯核函数  sigmoid：Sigmoid型核函数

class SVM:
    def __init__(self, manum, minum, maxtimes=800, lin1_C=0.09, ploy_d=10.3, rbf_sigma=0.15, tanh_beta=0.9, tanh_theta=-0.6,
                 kernel='rbf', epsilon=0.5):  # 参数均为float形式

        self.maxtimes = maxtimes  # 循环最大次数

        self.lin1_C = lin1_C  # 线性可分问题软间隔
        self.ploy_d = ploy_d   # 多项式核函数参数
        self.rbf_sigma = rbf_sigma   # 高斯核函数参数
        self.tanh_beta = tanh_beta   # Sigmoid型核函数参数
        self.tanh_theta = tanh_theta

        self.kernel = kernel  # 用到的核函数

        self.epsilon = epsilon

        self.manum = manum
        self.minum = minum

    # 数据还原尺度
    def reyuan(self, x):
        x = x * (self.manum - self.minum) + self.minum
        return  x

    #  训练函数
    def train_svm(self, shuxing, biaoqian, ceshisx):

        # 创建会话
        sess = tf.Session()

        # 训练数据占位符
        x_data = tf.placeholder(shape=[None, len(shuxing[0])], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # 预测数据占位符
        prexdata = tf.placeholder(shape=[None, len(shuxing[0])], dtype=tf.float32)

        # 拉格朗日因子变量
        Lagrange = tf.Variable(tf.random_normal(shape=[1, len(shuxing)]), dtype=tf.float32)  # 和样本的个数是一致的

        Lagrange_Star = tf.Variable(tf.random_normal(shape=[1, len(shuxing)]), dtype=tf.float32)  # 和样本的个数是一致的


        # linear 线性核函数
        if self.kernel == 'linear':
            # 计算核函数值
            kernel_num = tf.matmul(x_data, tf.transpose(x_data))
            # 预测函数
            pred_num = tf.matmul(x_data, tf.transpose(prexdata))

        elif self.kernel == 'poly':
            # 计算核函数值
            kernel_num = tf.pow(tf.matmul(x_data, tf.transpose(x_data)), self.ploy_d)
            # 预测函数
            pred_num = tf.pow(tf.matmul(x_data, tf.transpose(prexdata)), self.ploy_d)

        elif self.kernel == 'sigmoid':
            # 计算核函数值
            kernel_num = tf.tanh(self.tanh_beta * tf.matmul(x_data, tf.transpose(x_data)) + self.tanh_theta)
            # 预测函数
            pred_num = tf.tanh(self.tanh_beta * tf.matmul(x_data, tf.transpose(prexdata)) + self.tanh_theta)

        elif self.kernel == 'rbf':
            # 计算核函数的值，将模的平方展开：a方+b方-2ab
            xdatafang = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
            momo = tf.add(tf.subtract(xdatafang, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),
                            tf.transpose(xdatafang))
            kernel_num = tf.exp(tf.multiply((1/(-2 * tf.pow(self.rbf_sigma, 2))), tf.abs(momo)))

            # 计算预测函数的值，将模的平方展开：a方+b方-2ab
            xfang = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
            prefang = tf.reshape(tf.reduce_sum(tf.square(prexdata), 1), [-1, 1])
            mofang = tf.add(tf.subtract(xfang, tf.multiply(2., tf.matmul(x_data, tf.transpose(prexdata)))),
                            tf.transpose(prefang))
            pred_num = tf.exp(tf.multiply((1/(-2 * tf.pow(self.rbf_sigma, 2))), tf.abs(mofang)))
        else:
            print('核函数命名错误')
            kernel_num = 0
            pred_num = 0
            import time
            time.sleep(1000)

        # 计算成本函数
        # 第一项拉格朗日因子的和
        sum_alpha = - self.epsilon * (tf.reduce_sum(Lagrange) + tf.reduce_sum(Lagrange_Star))
        # 第二项拉格朗日因子的差与y值的乘积之和
        ypro = tf.reduce_sum(tf.matmul(tf.subtract(Lagrange, Lagrange_Star), y_target,))
        # 第三项
        la_la = tf.matmul(tf.transpose(tf.subtract(Lagrange, Lagrange_Star)), tf.subtract(Lagrange, Lagrange_Star))

        second_term = -0.5 * tf.reduce_sum(tf.multiply(kernel_num, la_la))
        # 最终的
        loss = tf.add(tf.add(sum_alpha, second_term), ypro)

        # 计算预测的数值
        predition_num = tf.reduce_sum(tf.matmul(tf.subtract(Lagrange, Lagrange_Star), pred_num))
        # 将数据还原为原始的尺度
        predition = self.reyuan(predition_num)
        yydata = self.reyuan(y_target)

        # 计算误差
        error = tf.square(predition - yydata)


        # 调用优化器
        my_opt = tf.train.GradientDescentOptimizer(0.9)  # 学习率
        train_step = my_opt.minimize(loss)

        # 初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)

        # 开始训练
        loss_vec = []  # 存储每一次的误差

        # 存储训练时候的最小二乘法的误差
        erc_train = []
        erc_pre = []

        #  属性数据shape = (样本数，单个样本特征数)
        #  标签数据shape = (样本数，1)
        labely = biaoqian.reshape(-1, 1)  # 更改维度
        for i in range(self.maxtimes):
            # 训练
            sess.run(train_step, feed_dict={x_data: shuxing, y_target: labely})  # 全部样本一齐训练
            # 获得误差
            temp_loss = sess.run(loss, feed_dict={x_data: shuxing, y_target: labely})
            loss_vec.append(temp_loss)

            # 输出最小二乘法的误差
            # 训练数据
            trlas = sess.run(error, feed_dict={x_data: shuxing, y_target: labely, prexdata: shuxing})
            # 预测数据
            prelas = sess.run(error, feed_dict={x_data: shuxing, y_target: labely, prexdata: ceshisx})

            erc_train.append(trlas)
            erc_pre.append(prelas)

        # 返回网络的预测值
        nettr = sess.run(predition, feed_dict={x_data: shuxing, y_target: labely, prexdata: shuxing})

        netpre = sess.run(predition, feed_dict={x_data: shuxing, y_target: labely, prexdata: ceshisx})


        # 返回训练误差，训练二乘误差，预测二乘误差
        return loss_vec, erc_train, erc_pre, nettr, netpre


#  数据
def result(data, he):
    #  训练数据
    xd = data[0]
    yd = data[1].T[0]
    #  测试数据
    texd = data[2]

    print(yd)

    # 建立模型
    resu = SVM(kernel=he, manum=data[4][0], minum=data[4][1])
    # 开始训练
    lo, eetr, eepr, nettrr, netpree = resu.train_svm(xd, yd, texd)

    print(lo)
    print(eetr)
    print(eepr)
    print(nettrr)

    print(netpree)


    # 绘制成本函数曲线，以及每一次训练的训练的二乘误差和预测数据的二乘误差
    fig, host = plt.subplots()
    # 用来控制多y轴
    par1 = host.twinx()
    par2 = host.twinx()
    #  多条曲线
    p1, = host.plot(list(range(len(lo))), lo, "b-", marker='8', label='成本函数的误差', linewidth=2)

    p2, = par1.plot(list(range(len(eetr))), eetr, "b--", marker='*', label='训练数据的最小二乘误差', linewidth=2)

    p3, = par2.plot(list(range(len(eepr))), eepr, "r-", marker='8', label='预测数据的最小二乘误差', linewidth=2)

    #  每个轴的内容

    host.set_xlabel("成本函数")
    par1.set_ylabel("准确率")
    par2.set_ylabel("准确率")

    #  控制每个y轴内容的颜色
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())


    #  控制每个Y轴刻度数字的颜色以及线粗细
    tkw = dict(size=6, width=3)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)

    #  添加图例
    lines = [p1, p2, p3]
    host.legend(lines, [l.get_label() for l in lines], loc='best')

    #  添加标题
    plt.title('北京Pm2.5回归 核函数：%s' % he)

    #  控制每个Y轴刻度线的颜色
    ax = plt.gca()
    ax.spines['left'].set_color('blue')
    ax.spines['right'].set_color('red')

    # 显示图片
    plt.show()

    return nettrr, netpree

# 绘图的函数
def huitu(suout, shiout, c=['b', 'k'], sign='训练', cudu=3):
    print(suout)
    print(shiout)
    # 绘制原始数据和预测数据的对比
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(suout))), suout, c=c[0], linewidth=cudu, label='%s：算法输出' % sign)
    plt.plot(list(range(len(shiout))), shiout, c=c[1], linewidth=cudu, label='%s：实际值' % sign)
    plt.legend()
    plt.title('对比')

    # 绘制误差和0的对比图
    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(suout))), suout - shiout, c='r', linewidth=cudu, label='%s：误差' % sign)
    plt.plot(list(range(len(suout))), list(np.zeros(len(suout))), c='r', linewidth=cudu, label='0值')
    plt.legend()
    plt.title('误差')
    # 需要添加一个误差的分布图

    # 显示
    plt.show()


'''第四部分：最终的运行程序'''
if __name__ == "__main__":
    datasvr = rdata.model_data
    outtri, poupre = result(datasvr, he='rbf')

    trii = datasvr[1].T[0] * (datasvr[4][0] - datasvr[4][1]) + datasvr[4][1]
    huitu(trii, outtri, c=['b', 'k'], sign='训练', cudu=3)

    prii = datasvr[3].T[0] * (datasvr[4][0] - datasvr[4][1]) + datasvr[4][1]
    huitu(prii, poupre, c=['b', 'k'], sign='预测', cudu=3)







