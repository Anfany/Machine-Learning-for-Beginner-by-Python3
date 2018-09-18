# -*- coding：utf-8 -*-
# &Author AnFany

# 基于TensorFlow实现支持向量机回归
# 防止不稳定，不采用batchsize的方式训练，每一次训练都是全部样本
# 因此不适用于样本数据量较大的情况
# 引入惩罚项实现对拉格朗日因子的限制

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
    def __init__(self, manum, minum, maxtimes=20000, C=0.09, ploy_d=6, rbf_sigma=0.15, tanh_beta=0.9, tanh_theta=-0.6,
                 kernel='rbf', epslion=0.5, beta=0, quj=100, ling=100):  # 参数均为float形式

        self.maxtimes = maxtimes  # 循环最大次数

        self.C = C  # 软间隔
        self.ploy_d = ploy_d   # 多项式核函数参数
        self.rbf_sigma = rbf_sigma   # 高斯核函数参数
        self.tanh_beta = tanh_beta   # Sigmoid型核函数参数
        self.tanh_theta = tanh_theta

        self.kernel = kernel  # 用到的核函数

        self.epslion = epslion
        self.beta = beta

        self.manum = manum
        self.minum = minum

        self.quj = quj
        self.ling = ling

    # 数据还原尺度
    def reyuan(self, x):
        hxx = x * (self.manum - self.minum) + self.minum
        return hxx

    #  训练函数
    def train_svm(self, shuxing, biaoqian, ceshisx, ceshibq):

        # 创建会话
        sess = tf.Session()

        # 训练数据占位符
        x_data = tf.placeholder(shape=[None, len(shuxing[0])], dtype=tf.float32)
        y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)

        # 预测数据占位符
        prexdata = tf.placeholder(shape=[None, len(shuxing[0])], dtype=tf.float32)


        # 线性回归
        if self.kernel == 'lin1':
            # 回归线的变量
            W = tf.Variable(tf.random_normal(shape=[len(shuxing[0]), 1]), dtype=tf.float32)
            b = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)

            # 回归线值
            model_output = tf.transpose(tf.subtract(tf.matmul(x_data, W), b))

            # L2范数
            l2_term = tf.reduce_sum(tf.square(W))

            # 最终的成本函数
            loss = tf.reduce_mean(
                tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), self.epslion))) + self.beta * l2_term

            # 预测的函数
            predition_num = tf.transpose(tf.subtract(tf.matmul(prexdata, W), b))


        else:
            # 拉格朗日因子变量
            Lagrange = tf.Variable(tf.random_normal(shape=[1, len(shuxing)]), dtype=tf.float32)  # 和样本的个数是一致的

            Lagrange_Star = tf.Variable(tf.random_normal(shape=[1, len(shuxing)]), dtype=tf.float32)  # 和样本的个数是一致的

            b = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)


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


            # 计算成本函数
            # 第一项
            la_la = tf.matmul(tf.transpose(tf.subtract(Lagrange, Lagrange_Star)), tf.subtract(Lagrange, Lagrange_Star))
            first = -0.5 * tf.reduce_sum(tf.multiply(kernel_num, la_la))

            # 第二项
            second = -self.epslion * (tf.reduce_sum(Lagrange) + tf.reduce_sum(Lagrange_Star))

            # 第三项
            third = tf.reduce_sum(tf.multiply(tf.subtract(Lagrange, Lagrange_Star), y_target))

            # 成本函数
            sonloss = tf.negative(tf.add(tf.add(first, second), third))

            # 因为拉格朗日变量是由限制的，在tensorflow中，利用惩罚项来实现对变量取值的控制,
            # 通过加大惩罚项的权重来加快舍弃不符合的拉格朗日因子
            #  首先是所有的拉格朗日应该在[0, C] 之间

            # 首先是针对负数
            qujian_fu = tf.reduce_sum(tf.nn.relu(tf.negative(Lagrange))) + tf.reduce_sum(tf.nn.relu(tf.negative(Lagrange_Star)))

            # 针对超C值
            qujian_c = tf.reduce_sum(tf.nn.relu(tf.subtract(tf.abs(Lagrange), self.C))) + \
                       tf.reduce_sum(tf.nn.relu(tf.subtract(tf.abs(Lagrange_Star), self.C)))

            # 结合上面两项
            qujian = self.quj * tf.add(qujian_c, qujian_fu)

            # 再者是样本对应的拉格朗日因子的差的和是0
            cha = tf.abs(tf.reduce_sum(tf.subtract(Lagrange, Lagrange_Star)))
            sub = self.ling * cha


            # 带有变量限制的拉格朗日因子总体的成本
            loss = tf.add(tf.add(sonloss, qujian), sub)

            # 计算预测的数值
            predition_num = tf.reduce_sum(tf.multiply(tf.transpose(tf.subtract(Lagrange, Lagrange_Star)), pred_num), 0)

        # 将数据还原为原始的尺度
        predition = self.reyuan(predition_num)
        yydata = self.reyuan(y_target)

        # 计算误差
        error = tf.reduce_sum(tf.square(predition - yydata))


        # 调用优化器
        my_opt = tf.train.GradientDescentOptimizer(0.000005)  # 学习率
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
        for i in range(self.maxtimes):
            # 训练
            sess.run(train_step, feed_dict={x_data: shuxing, y_target: biaoqian, prexdata: shuxing})  # 全部样本一齐训练

            # 获得误差
            temp_loss = sess.run(loss, feed_dict={x_data: shuxing, y_target: biaoqian, prexdata: shuxing})
            loss_vec.append(temp_loss)


            # 输出最小二乘法的误差
            # 训练数据
            trlas = sess.run(error, feed_dict={x_data: shuxing, y_target: biaoqian, prexdata: shuxing})
            # 预测数据
            prelas = sess.run(error, feed_dict={x_data: shuxing, y_target: ceshibq, prexdata: ceshisx})

            erc_train.append(trlas)
            erc_pre.append(prelas)


            print(temp_loss)


            chahzi, chaha, lii = sess.run([qujian_fu, qujian_c, cha])

            print('负数：%.4f' % chahzi, '超C：%.4f' % chaha, '和值：%.4f' % lii)


            if temp_loss < 1:
                break




        # 返回网络的预测值
        nettr = sess.run(predition, feed_dict={x_data: shuxing, prexdata: shuxing})

        netpre = sess.run(predition, feed_dict={x_data: shuxing, prexdata: ceshisx})


        du = sess.run(Lagrange_Star)

        fu = sess.run(Lagrange)

        print(fu)

        print(du)


        # 返回训练误差，训练二乘误差，预测二乘误差
        return loss_vec, erc_train, erc_pre, nettr, netpre


#  数据
def result(data, he):
    #  训练数据
    xd = data[0]
    yd = data[1]
    #  测试数据
    texd = data[2]
    teyd = data[3]

    # 建立模型
    resu = SVM(kernel=he, manum=data[4][0], minum=data[4][1])
    # 开始训练
    lo, eetr, eepr, nettrr, netpree = resu.train_svm(xd, yd, texd, teyd)


    # 绘制成本函数曲线，以及每一次训练的训练的二乘误差和预测数据的二乘误差
    fig, host = plt.subplots()
    # 用来控制多y轴
    par1 = host.twinx()
    #  多条曲线
    p1, = host.plot(list(range(len(eetr))), eetr, "b*", marker='*', linewidth=2, label='训练')

    p2, = par1.plot(list(range(len(eepr))), eepr, "r--", marker='8', linewidth=2, label='预测')

    #  每个轴的内容

    host.set_ylabel("误差")
    par1.set_ylabel("误差")
    host.set_xlabel('训练的次数')


    #  控制每个y轴内容的颜色
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())



    #  控制每个Y轴刻度数字的颜色以及线粗细
    tkw = dict(size=6, width=3)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)


    #  添加图例
    lines = [p1, p2]
    host.legend(lines, [l.get_label() for l in lines], loc='best')

    #  添加标题
    plt.title('北京Pm2.5回归 方法：%s' % he)

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
    plt.plot(list(range(len(suout))), suout, c=c[0], linewidth=cudu, label='%s：算法' % sign)
    plt.plot(list(range(len(shiout))), shiout, c=c[1], linewidth=cudu, label='%s：实际' % sign)
    plt.legend()
    plt.title('真实与算法输出数据对比')

    # 绘制误差和0的对比图
    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(suout))), suout - shiout, c='r', linewidth=cudu, label='%s：算法-实际' % sign)
    plt.plot(list(range(len(suout))), list(np.zeros(len(suout))), c='r', linewidth=cudu, label='0值')
    plt.legend()
    plt.title('误差 VS 0')
    # 需要添加一个误差的分布图

    # 显示
    plt.show()


'''第四部分：最终的运行程序'''
if __name__ == "__main__":
    datasvr = rdata.model_data
    outtri, poupre = result(datasvr, he='rbf')

    trii = datasvr[1][0] * (datasvr[4][0] - datasvr[4][1]) + datasvr[4][1]
    huitu(trii, outtri, c=['b', 'k'], sign='训练', cudu=1.5)

    prii = datasvr[3][0] * (datasvr[4][0] - datasvr[4][1]) + datasvr[4][1]
    huitu(prii, poupre, c=['b', 'k'], sign='预测', cudu=1.5)
