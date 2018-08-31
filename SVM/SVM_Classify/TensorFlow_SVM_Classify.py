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

# 引入数据
import SVM_Classify_Data as sdata

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
    def __init__(self, maxtimes=800, lin1_C=0.09, ploy_d=10.3, rbf_sigma=0.15, tanh_beta=0.9, tanh_theta=-0.6,
                 kernel='rbf'):  # 参数均为float形式

        self.maxtimes = maxtimes  # 循环最大次数

        self.lin1_C = lin1_C  # 线性可分问题软间隔
        self.ploy_d = ploy_d   # 多项式核函数参数
        self.rbf_sigma = rbf_sigma   # 高斯核函数参数
        self.tanh_beta = tanh_beta   # Sigmoid型核函数参数
        self.tanh_theta = tanh_theta

        self.kernel = kernel  # 用到的核函数

    #  训练函数
    def train_svm(self, shuxing, biaoqian, ceshisx):

        # 创建会话
        sess = tf.Session()

        # 训练数据占位符
        x_data = tf.placeholder(shape=[None, len(shuxing[0])], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # 预测数据占位符
        prexdata = tf.placeholder(shape=[None, len(shuxing[0])], dtype=tf.float32)

        # 线性可分情况
        if self.kernel == 'lin1':
            # 线性可分：变量
            W = tf.Variable(tf.random_normal(shape=[len(shuxing[0]), 1]), dtype=tf.float32)
            b = tf.Variable(tf.random_normal(shape=[1, 1]), dtype=tf.float32)

            # 分割线的值
            model_output = tf.subtract(tf.matmul(x_data, W), b)

            # L2范数
            l2_term = tf.reduce_sum(tf.square(W))

            # 成本函数
            class_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
            loss = tf.add(class_term, tf.multiply(self.lin1_C, l2_term))

            # 计算预测的值
            prediction = tf.squeeze(tf.sign(tf.subtract(tf.matmul(prexdata, W), b)))

        # 核函数情况
        else:
            # 其实就是拉格朗日因子变量
            Lagrange = tf.Variable(tf.random_normal(shape=[1, len(shuxing)]))  # 和样本的个数是一致的

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
            sum_alpha = tf.reduce_sum(Lagrange)
            # 计算第二项
            la_la = tf.matmul(tf.transpose(Lagrange), Lagrange)
            y_y = tf.matmul(y_target, tf.transpose(y_target))
            second_term = tf.reduce_sum(tf.multiply(kernel_num, tf.multiply(la_la, y_y)))
            # 最终的
            loss = tf.negative(tf.subtract(sum_alpha, second_term))

            # 计算预测的类别以及正确率
            prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), Lagrange), pred_num)
            # 类别输出，shape=（样本数,）
            prediction = tf.squeeze(tf.sign(prediction_output))


        # 调用优化器
        my_opt = tf.train.GradientDescentOptimizer(0.004)  # 学习率
        train_step = my_opt.minimize(loss)

        # 初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)

        # 开始训练
        loss_vec = []  # 存储每一次的误差

        #  属性数据shape = (样本数，单个样本特征数)
        #  标签数据shape = (样本数，1)
        labely = biaoqian.reshape(-1, 1)  # 更改维度
        for i in range(self.maxtimes):
            # 训练
            sess.run(train_step, feed_dict={x_data: shuxing, y_target: labely})  # 全部样本一齐训练
            # 获得误差
            temp_loss = sess.run(loss, feed_dict={x_data: shuxing, y_target: labely})
            loss_vec.append(temp_loss)

            # 获得拉格朗日因子的值
            # 因为当kenerl的值为lin1的时候，没有定义拉格朗日
            try:
                lan = Lagrange.eval(session=sess)
            except UnboundLocalError:
                pass

        # 输出预测的类别
        # 训练数据
        trlas = sess.run(prediction, feed_dict={x_data: shuxing, y_target: labely, prexdata: shuxing})
        # 预测数据
        prelas = sess.run(prediction, feed_dict={x_data: shuxing, y_target: labely, prexdata: ceshisx})


        # 返回训练误差，训练输出，预测输出
        if self.kernel != 'lin1':
            return loss_vec, trlas, prelas, np.array(lan)[0]
        else:
            return loss_vec, trlas, prelas

    # 计算正确率的函数
    def acc(self, reallabel, netlabel):  # shape=(样本数，)
        accua = np.array(reallabel)[np.array(reallabel) == np.array(netlabel)]
        return len(accua) / len(netlabel)

#  K折数据集字典
def result(datadict, he):
    sign = []
    trainacc, testacc, vec = [], [], []
    resu = []
    for jj in datadict:
        #  训练数据
        xd = datadict[jj][0][:, :-1]
        yd = datadict[jj][0][:, -1]
        #  测试数据
        texd = datadict[jj][1][:, :-1]
        teyd = datadict[jj][1][:, -1]

        # 建立模型
        resu = SVM(kernel=he)
        # 开始训练
        lo, tracu, teacu, *vector = resu.train_svm(xd, yd, texd)
        # 训练完，储存训练、测试的精确度结果
        trainacc.append(resu.acc(yd, tracu))
        testacc.append(resu.acc(teyd, teacu))
        if he != 'lin1':
            vec.append(len(np.array(vector)[np.array(vector) < 0]))
        else:
            vec.append(0)
        sign.append(jj)

    # 绘制多y轴图
    fig, host = plt.subplots()
    # 用来控制多y轴
    par1 = host.twinx()
    #  多条曲线
    p1, = host.plot(sign, trainacc, "b-", marker='8', label='训练', linewidth=2)
    pp, = host.plot(sign, testacc, "b--", marker='*', label='测试', linewidth=2)
    if he == 'lin1':
        p2, = par1.plot(sign, vec, "r-", marker='8', label='支持向量[未定义]', linewidth=2)
    else:
        p2, = par1.plot(sign, vec, "r-", marker='8', label='支持向量', linewidth=2)
    #  每个轴的内容
    host.set_xlabel("K折数据集")
    host.set_ylabel("分类准确率")
    par1.set_ylabel("个数")
    #  控制每个y轴内容的颜色
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    #  控制每个Y轴刻度数字的颜色以及线粗细
    tkw = dict(size=6, width=3)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)

    #  添加图例
    lines = [p1, pp, p2]
    host.legend(lines, [l.get_label() for l in lines], loc='lower center')

    #  添加标题
    plt.title('K折心脏病数据集SVM分类结果对比 核函数：%s' % he)

    #  控制每个Y轴刻度线的颜色
    ax = plt.gca()
    ax.spines['left'].set_color('blue')
    ax.spines['right'].set_color('red')

    # 显示图片
    plt.show()


'''第四部分：最终的运行程序'''
if __name__ == "__main__":
    result(sdata.kfold_train_datadict, 'rbf')



