# -*- coding：utf-8 -*-
# &Author  AnFany

'''第一部分：库'''

import BPNN_Classify_Data as bpd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False # 显示负号

#  分类数
countclass = 2


'''第二部分：函数'''

# 隐层的激活函数
# 注意：程序中激活函数中的导函数中的输入，是执行过激活函数后的值
def Sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def Sigmoid_der(s):
    y = np.multiply(s, 1 - s)
    return y


def Relu(x):
    s = np.maximum(0, x)
    return s
def Relu_der(s):
    y = np.where(s > 0, np.ones(s.shape), np.zeros(s.shape))
    return y


def Tanh(x):
    s = np.tanh(x)
    return s
def Tanh_der(s):
    y = 1 - np.multiply(s, s)
    return y


# 成本函数
def cross_entropy(yreal, yout):  # yreal 真实值 yout 网络输出值
    costnum = - (yreal * np.log(yout) + (1 - yreal) * np.log(1 - yout)).sum() / len(yreal)
    return costnum
def cross_entropy_der(yreal, yout, num=1e-8):
    return -(yreal - yout) / (yout * (1 - yout)) / len(yreal)


# 输出层的激活函数
def Linear(x):  # 线性函数，将数据的范围平移··到输出数据的范围
    return x
def Linear_der(s):
    y = np.zeros(shape=s.shape)
    return y


#  根据输出的结果判断类别的函数
def judge(ydata):
    maxnum = np.max(ydata, axis=1)
    lastdata = []
    for ii in range(len(ydata)):
        maxindex = list(ydata[ii]).index(maxnum[ii])
        fu = [0] * len(ydata[0])
        fu[maxindex] = 1
        lastdata.append(fu)
    return np.array(lastdata)


#  根据输出的结果以及真实结果输出分类起的效果
def outvsreal(outdata, realdata):
    subdata = outdata - realdata
    sundata = np.sum(np.abs(subdata), axis=1)
    correct = list(sundata).count(0)
    return correct / len(outdata)


'''第三部分：实现神经网络'''


class BPNN():
    def __init__(self, train_in, train_out, add_in, add_out, learn_rate=0.03, son_samples=50 \
                 , iter_times=200000, hidden_layer=[100, 100, 100], middle_name='Sigmoid' \
                 , last_name='Sigmoid', cost_func='cross_entropy', norr=0.00002, break_error=0.84):
        self.train_in = train_in  # 每一行是一个样本输入
        self.train_out = train_out  # 每一行是一个样本输出

        self.add_in = add_in
        self.add_out = add_out

        self.learn_rate = learn_rate  # 学习率
        self.son_samples = son_samples  # 子样本个数
        self.iter_times = iter_times  # 迭代次数

        self.all_layer = [len(self.train_in[0])] + hidden_layer + [len(self.train_out[0])]  # 定义各层的神经元个数
        self.func_name = [middle_name] * len(hidden_layer) + [last_name]  # 定义各层的激活函数

        #  参数设置
        # self.weight = [np.array(np.random.randn(x, y) / np.sqrt(x) * 0.01, dtype=np.float64) \
        #                for x, y in zip(self.all_layer[:-1], self.all_layer[1:])]  # 初始化权重

        self.weight = [np.array(np.random.randn(x, y)) for x, y in
                       zip(self.all_layer[:-1], self.all_layer[1:])]  # 初始化权重

        self.bias = [np.random.randn(1, y) * 0.01 for y in self.all_layer[1:]]  # 初始化偏置

        self.cost_func = cost_func
        self.norr = norr
        self.break_error = break_error

    # 梯度下降法
    def train_gadient(self):
        # 采用随机小批量梯度下降
        # 首先结合输入和输出数据
        alldata = np.hstack((self.train_in, self.train_out))
        np.random.shuffle(alldata)  # 打乱顺序
        # 输入和输出分开
        trin = alldata[:, :len(self.train_in[0])]  # 每一行是一个样本输入
        trout = alldata[:, len(self.train_in[0]):]  # 每一行是一个样本输出

        # 计算批次数
        pici = int(len(alldata) / self.son_samples) + 1

        # 存储误差值
        error_list = []
        # 存储误差值
        error_list_add =[]
        # 存储训练数据集合的正确率
        xlcorr = []
        # 存储验证数据集合的正确率
        adcorr = []
        # 存储权重和偏置的字典
        sacewei = []
        iter = 0

        while iter < self.iter_times:
            for times in range(pici):
                in_train = trin[times * self.son_samples: (times + 1) * self.son_samples, :]
                out_train = trout[times * self.son_samples: (times + 1) * self.son_samples, :]
                # 开始步入神经网络

                a = list(range(len(self.all_layer)))  # 储存和值
                z = list(range(len(self.all_layer)))  # 储存激活值

                a[0] = in_train.copy()  # 和值
                z[0] = in_train.copy()  # 激活值

                # 开始逐层正向传播
                for forward in range(1, len(self.all_layer)):  # 1,2,3
                    a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                    z[forward] = eval(self.func_name[forward - 1])(a[forward])

                # 定义输出层误差
                ne = list(range(len(self.all_layer)))  # 储存输出的误差值

                qiaia = eval(self.cost_func + '_der')(out_train, z[-1])
                hhou = eval(self.func_name[-1] + '_der')(z[-1])
                ne[-1] = np.multiply(qiaia, hhou)

                # 开始逐层反向传播
                for backward in range(len(self.all_layer) - 1, 0, -1):
                    qianzhe = np.dot(ne[backward], self.weight[backward - 1].T)
                    houzhe = eval(self.func_name[backward - 1] + '_der')(z[backward - 1])
                    ne[backward - 1] = np.multiply(qianzhe, houzhe)

                # 开始逐层计算更改W和B值
                dw = list(range(len(self.all_layer) - 1))
                db = list(range(len(self.all_layer) - 1))

                # L2正则化
                for iwb in range(len(self.all_layer) - 1):
                    dw[iwb] = np.dot(a[iwb].T, ne[iwb + 1]) / self.son_samples + \
                              (self.norr / self.son_samples) * dw[iwb]

                    db[iwb] = np.sum(ne[iwb + 1], axis=0, keepdims=True) / self.son_samples + \
                              (self.norr / self.son_samples) * db[iwb]

                # 更改权重
                for ich in range(len(self.all_layer) - 1):
                    self.weight[ich] -= self.learn_rate * dw[ich]
                    self.bias[ich] -= self.learn_rate * db[ich]

            # 整个样本迭代一次计算训练样本的误差
            a[0] = trin.copy()  # 和值
            z[0] = trin.copy()  # 激活值
            for forward in range(1, len(self.all_layer)):
                a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                z[forward] = eval(self.func_name[forward - 1])(a[forward])

            # 打印训练数据误差值
            errortrain = eval(self.cost_func)(trout, z[-1])
            train_corr = outvsreal(judge(z[-1]), trout)
            print('第%s代训练样本误差：%.9f, 分类的正确率为%.5f' % (iter, errortrain, train_corr))
            error_list.append(errortrain)

            # 整个样本迭代一次计算验证样本的误差
            a[0] = self.add_in.copy()  # 和值
            z[0] = self.add_in.copy()  # 激活值
            for forward in range(1, len(self.all_layer)):
                a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                z[forward] = eval(self.func_name[forward - 1])(a[forward])

            # 打印验证数据误差值
            erain = eval(self.cost_func)(self.add_out, z[-1])
            add_correct = outvsreal(judge(z[-1]), self.add_out)
            print('-----------------------验证样本误差：%.9f, 分类的正确率为%.5f' % (erain, add_correct))
            error_list_add.append(erain)

            iter += 1
            xlcorr.append(train_corr)
            adcorr.append(add_correct)

            # 存储权重和偏置的字典
            if len(sacewei) == 4:
                sacewei = sacewei[1:].copy()
                sacewei.append([self.weight, self.bias])
            else:
                sacewei.append([self.weight, self.bias])

            #  提前结束的判断(验证数据集分类正确率连续下降三次，退出循环, 并且存储第一次下降前的权重和偏置）
            if len(adcorr) >= 4:
                # 判断连续三次下降
                edlist = adcorr[-4:-1]
                delist = adcorr[-3:]
                sublist = np.array(edlist) - np.array(delist)
                if np.all(sublist > 0):
                    self.weight, self.bias = sacewei[0]
                    break

        return self.weight, self.bias, error_list, xlcorr, adcorr, error_list_add

    def train_adam(self, mom=0.9, prop=0.9):
        # 采用随机小批量梯度下降
        # 首先结合输入和输出数据
        alldata = np.hstack((self.train_in, self.train_out))
        np.random.shuffle(alldata)  # 打乱顺序
        # 输入和输出分开
        trin = alldata[:, :len(self.train_in[0])]  # 每一行是一个样本输入
        trout = alldata[:, len(self.train_in[0]):]  # 每一行是一个样本输出

        # 计算批次数
        pici = int(len(alldata) / self.son_samples) + 1

        # 存储误差值
        error_list = []
        iter = 0

        while iter < self.iter_times:
            for times in range(pici):
                in_train = trin[times * self.son_samples: (times + 1) * self.son_samples, :]
                out_train = trout[times * self.son_samples: (times + 1) * self.son_samples, :]
                # 开始步入神经网络

                a = list(range(len(self.all_layer)))  # 储存和值
                z = list(range(len(self.all_layer)))  # 储存激活值

                a[0] = in_train.copy()  # 和值
                z[0] = in_train.copy()  # 激活值

                # 开始逐层正向传播
                for forward in range(1, len(self.all_layer)):  # 1,2,3
                    a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                    z[forward] = eval(self.func_name[forward - 1])(a[forward])

                # 定义输出层误差
                ne = list(range(len(self.all_layer)))  # 储存输出的误差值

                qiaia = eval(self.cost_func + '_der')(out_train, z[-1])
                hhou = eval(self.func_name[-1] + '_der')(z[-1])
                ne[-1] = np.multiply(qiaia, hhou)

                # 开始逐层反向传播
                for backward in range(len(self.all_layer) - 1, 0, -1):
                    qianzhe = np.dot(ne[backward], self.weight[backward - 1].T)
                    houzhe = eval(self.func_name[backward - 1] + '_der')(z[backward - 1])
                    ne[backward - 1] = np.multiply(qianzhe, houzhe)

                # 开始逐层计算更改W和B值
                dw = list(range(len(self.all_layer) - 1))
                db = list(range(len(self.all_layer) - 1))

                # L2正则化
                for iwb in range(len(self.all_layer) - 1):
                    dw[iwb] = np.dot(a[iwb].T, ne[iwb + 1]) / self.son_samples + \
                              (self.norr / self.son_samples) * dw[iwb]

                    db[iwb] = np.sum(ne[iwb + 1], axis=0, keepdims=True) / self.son_samples + \
                              (self.norr / self.son_samples) * db[iwb]

                try:
                    for im in range(len(self.all_layer) - 1):
                        vdw[im] = mom * vdw[im] + (1 - mom) * dw[im]
                        vdb[im] = mom * vdb[im] + (1 - mom) * db[im]

                        sdw[im] = mom * sdw[im] + (1 - mom) * (dw[im] ** 2)
                        sdb[im] = mom * sdb[im] + (1 - mom) * (db[im] ** 2)
                except NameError:
                    vdw = [np.zeros(w.shape) for w in self.weight]
                    vdb = [np.zeros(b.shape) for b in self.bias]

                    sdw = [np.zeros(w.shape) for w in self.weight]
                    sdb = [np.zeros(b.shape) for b in self.bias]

                    for im in range(len(self.all_layer) - 1):
                        vdw[im] = (1 - mom) * dw[im]
                        vdb[im] = (1 - mom) * db[im]

                        sdw[im] = (1 - prop) * (dw[im] ** 2)
                        sdb[im] = (1 - prop) * (db[im] ** 2)
                # 初始限制
                VDW = [np.zeros(w.shape) for w in self.weight]
                VDB = [np.zeros(b.shape) for b in self.bias]
                SDW = [np.zeros(w.shape) for w in self.weight]
                SDB = [np.zeros(b.shape) for b in self.bias]
                for slimit in range(len(self.all_layer) - 1):
                    VDW[slimit] = vdw[slimit] / (1 - mom ** (iter + 1))
                    VDB[slimit] = vdb[slimit] / (1 - mom ** (iter + 1))
                    SDW[slimit] = sdw[slimit] / (1 - prop ** (iter + 1))
                    SDB[slimit] = sdb[slimit] / (1 - prop ** (iter + 1))
                # 更改权重
                for ich in range(len(self.all_layer) - 1):
                    self.weight[ich] -= self.learn_rate * (VDW[ich] / (SDW[ich] ** 0.5 + 1e-8))
                    self.bias[ich] -= self.learn_rate * (VDB[ich] / (SDB[ich] ** 0.5 + 1e-8))

            # 整个样本迭代一次计算训练样本差
            a[0] = trin.copy()  # 和值
            z[0] = trin.copy()  # 激活值
            for forward in range(1, len(self.all_layer)):
                a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                z[forward] = eval(self.func_name[forward - 1])(a[forward])

            # 打印训练样本误差值
            errortrain = eval(self.cost_func)(trout, z[len(self.all_layer) - 1])
            print('第%s代总体误差：%.9f, 分类的正确率为%.5f' % (iter, errortrain, outvsreal(judge(z[-1]), trout)))

            error_list.append(errortrain)

            # 整个样本迭代一次计算验证样本的误差
            a[0] = self.add_in.copy()  # 和值
            z[0] = self.add_in.copy()  # 激活值
            for forward in range(1, len(self.all_layer)):
                a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                z[forward] = eval(self.func_name[forward - 1])(a[forward])

            # 打印验证数据误差值
            erain = eval(self.cost_func)(self.add_out, z[-1])
            add_correct = outvsreal(judge(z[-1]), self.add_out)
            print('-----------------验证样本误差：%.9f, 分类的正确率为%.5f' % (erain, add_correct))

            iter += 1

            #  提前结束的判断
            if add_correct > self.break_error:
                break
        return self.weight, self.bias, error_list

    def predict(self, pre_in_data):
        pa = list(range(len(self.all_layer)))  # 储存和值
        pz = list(range(len(self.all_layer)))  # 储存激活值
        pa[0] = pre_in_data.copy()  # 和值
        pz[0] = pre_in_data.copy()  # 激活值
        for forward in range(1, len(self.all_layer)):
            pa[forward] = np.dot(pz[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
            pz[forward] = eval(self.func_name[forward - 1])(pa[forward])
        return pz[-1]


'''第四部分：数据'''
DDatadict = bpd.kfold_train_datadict


#  将数据分为输入数据以及输出数据
def divided(data, cgu):
    indata = data[:, :-cgu]
    outdata = data[:, -cgu:]
    return indata, outdata


#  将数据字典的值转化为训练输入，训练输出，验证输入、验证输出
def transall(listdata, count=countclass):
    trin, trout = divided(listdata[0], count)
    yanin, yanout = divided(listdata[1], count)
    return trin, trout, yanin, yanout


'''第五部分：最终的运行程序'''
if __name__ == "__main__":
    # 存储正确率 训练
    corrsave_train = []
    # 存储正确率 验证
    corrsave_add = []
    # 存储测试集合的正确率
    corrsave_test = []
    TEST_In, TEST_Out = divided(bpd.Test_data.values)
    # 开始K折交叉验证
    for fold in DDatadict:
        TRAIN_In, TRAIN_Out, ADD_In, ADD_Out = transall(DDatadict[fold])
        bpnn = BPNN(TRAIN_In, TRAIN_Out, ADD_In, ADD_Out)
        bpnn_train = bpnn.train_gadient()
        # 验证正确率
        test_outdata = bpnn.predict(TEST_In)
        teeee = outvsreal(judge(test_outdata), TEST_Out)
        print('第%s次验证：最终的测试数据集的正确率为%.4f' % (fold, teeee))
        # 存储K折训练、验证数据集的正确率
        corrsave_train.append(bpnn_train[3][-4])
        corrsave_add.append(bpnn_train[4][-4])
        corrsave_test.append(teeee)
        # 绘制训练数据集与验证数据集的正确率以及误差曲线
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('代数')
        ax1.set_ylabel('误差', color='r')
        plt.plot(list(range(len(bpnn_train[2]))), bpnn_train[2], label='训练', color='r', marker='*', linewidth=2)
        plt.plot(list(range(len(bpnn_train[5]))), bpnn_train[5], label='验证', color='r', marker='.', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='r')
        legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('#F0F8FF')
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('正确率', color='b')  # we already handled the x-label with ax1
        plt.plot(list(range(len(bpnn_train[3][:-3]))), bpnn_train[3][:-3], label='训练', color='b', marker='*', linewidth=2)
        plt.plot(list(range(len(bpnn_train[4][:-3]))), bpnn_train[4][:-3], label='验证', color='b', marker='.', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='b')
        legen = ax2.legend(loc='lower center', shadow=True, fontsize='x-large')
        legen.get_frame().set_facecolor('#FFFAFA')
        ax2.grid(True)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('%s折训练VS验证 结果对比' % fold, fontsize=16)
        plt.savefig(r'C:\Users\GWT9\Desktop\%s_foldui.jpg' % fold)

    # 绘制K次的结果展示
    plt.figure()
    plt.plot(list(range(len(corrsave_train))), corrsave_train, label='训练', color='b', marker='s', linewidth=2)
    plt.plot(list(range(len(corrsave_add))), corrsave_add, label='验证', color='r', marker='8', linewidth=2)
    plt.plot(list(range(len(corrsave_test))), corrsave_test, label='测试', color='k', marker='d', linewidth=2)
    plt.xlabel('折数')
    plt.ylabel('正确率')
    plt.title('绘制K次的不同数据集的结果展示', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig(r'C:\Users\GWT9\Desktop\last_foldui.jpg')
    plt.show()

