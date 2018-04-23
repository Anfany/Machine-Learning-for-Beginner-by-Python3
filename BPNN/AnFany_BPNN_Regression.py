#-*- coding：utf-8 -*-
# &Author  AnFany

# 适用于多维输出
from BPNN_DATA_Reg import model_data as R_data
import numpy as np

'''第一部分：数据准备'''
train_x_data = R_data[0]  # 训练输入
train_y_data = R_data[1]  # 训练输出

predict_x_data = R_data[2]  # 测试输入
predict_y_data = R_data[3]  # 测试输出

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
def Lsm(yreal, yout):
    costnum = (1 / 2) * np.sum((yreal - yout) ** 2) / len(yreal)
    return costnum
def Lsm_der(yreal, yout):  # 成本函数的导数为网络输出值减去真实值
    return yout - yreal



# 输出层的激活函数
def Linear(x):#线性函数，将数据的范围平移··到输出数据的范围
    return x

def Linear_der(s):
    y = np.zeros(shape=s.shape)
    return y


'''第三部分：实现神经网络'''

class BPNN():

    def __init__(self, train_in, train_out, learn_rate=0.03, son_samples=50, iter_times=10000, hidden_layer=[100,100],middle_name='Sigmoid', last_name='Sigmoid', cost_func='Lsm', norr=0.00002):
        self.train_in = train_in  # 每一行是一个样本输入
        self.train_out = train_out  # 每一行是一个样本输出

        self.learn_rate = learn_rate  # 学习率
        self.son_samples = son_samples  # 子样本个数
        self.iter_times = iter_times  # 迭代次数


        self.all_layer = [len(self.train_in[0])] + hidden_layer + [len(self.train_out[0])] # 定义各层的神经元个数
        self.func_name = [middle_name] * len(hidden_layer) + [last_name] #定义各层的激活函数


        #  参数设置
        # self.weight = [np.array(np.random.randn(x, y) / np.sqrt(x) * 0.01, dtype=np.float64) \
        #                for x, y in zip(self.all_layer[:-1], self.all_layer[1:])]  # 初始化权重

        self.weight = [np.array(np.random.randn(x, y)) for x, y in zip(self.all_layer[:-1], self.all_layer[1:])] #初始化权重

        self.bias = [np.random.randn(1, y) * 0.01 for y in self.all_layer[1:]]  # 初始化偏置


        self.cost_func = cost_func

        self.norr = norr

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
                for forward in range(1, len(self.all_layer)): # 1,2,3
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
                    dw[iwb] = np.dot(a[iwb].T, ne[iwb + 1]) / self.son_samples +\
                              (self.norr / self.son_samples) * dw[iwb]

                    db[iwb] = np.sum(ne[iwb + 1], axis=0, keepdims=True) / self.son_samples + \
                              (self.norr / self.son_samples) * db[iwb]

                # 更改权重
                for ich in range(len(self.all_layer) - 1):
                    self.weight[ich] -= self.learn_rate * dw[ich]
                    self.bias[ich] -= self.learn_rate * db[ich]

            # 整个样本迭代一次计算训练样本和测试样本的误差

            a[0] = trin.copy()  # 和值
            z[0] = trin.copy()  # 激活值
            for forward in range(1, len(self.all_layer)):
                a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                z[forward] = eval(self.func_name[forward - 1])(a[forward])

            # 打印误差值
            errortrain = eval(self.cost_func)(z[len(self.all_layer) - 1], trout)
            print('第%s代总体误差：%.9f' % (iter, errortrain))

            error_list.append(errortrain)

            iter += 1

            #  提前结束的判断
            if errortrain < 0.0004:
                break

        return self.weight, self.bias, error_list

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
                for forward in range(1, len(self.all_layer)): # 1,2,3
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
                    dw[iwb] = np.dot(a[iwb].T, ne[iwb + 1]) / self.son_samples +\
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

            # 整个样本迭代一次计算训练样本和测试样本的误差
            a[0] = trin.copy()  # 和值
            z[0] = trin.copy()  # 激活值
            for forward in range(1, len(self.all_layer)):
                a[forward] = np.dot(z[forward - 1], self.weight[forward - 1]) + self.bias[forward - 1]
                z[forward] = eval(self.func_name[forward - 1])(a[forward])

            # 打印误差值
            errortrain = eval(self.cost_func)(z[len(self.all_layer) - 1], trout)
            print('第%s代总体误差：%.9f' % (iter, errortrain))

            error_list.append(errortrain)

            iter += 1

            #  提前结束的判断
            if len(error_list) > 2 and error_list[-1] > error_list[-2]:
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


'''第四部分： 结果展示函数'''

import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 设置中文字体新宋体
#  绘制图像
def figure(real, net, le='训练'):
    length = len(real[0])
    # 绘制每个维度的对比图
    for iwe in range(length):
        plt.subplot(length, 1, iwe+1)
        plt.plot(list(range(len(real.T[iwe]))), real.T[iwe])
        plt.plot(list(range(len(net.T[iwe]))), net.T[iwe])
        plt.legend(['%s真实值'%le, '网络输出值'])
        if length == 1:
            plt.title('%s结果对比'%le)
        else:
            if iwe == 0:
                plt.title('%s结果对比: %s维度'%(le, iwe))
            else:
                plt.title('%s维度对比' %le)
    plt.show()

# 绘制成本函数曲线图
def costfig(errlist, le='成本函数曲线图'):
    plt.plot(list(range(len(errlist))), errlist)
    plt.title(le)
    plt.xlabel('迭代次数')
    plt.ylabel('成本函数值')
    plt.show()

#  因为训练数据较多，为了不影响展示效果，按序随机选取一定数量的数据，便于展示
def select(datax, datay, count=2000):
    sign = list(range(len(datax)))
    selectr_sign = np.random.choice(sign, count, replace=False)
    return datax[selectr_sign], datay[selectr_sign]

# 将输出的数据转换尺寸，变为原始数据的尺寸
def trans(ydata, minumber=R_data[4][0], maxumber=R_data[4][1]):
    return ydata * (maxumber - minumber) + minumber

'''第五部分：最终的运行程序'''
if __name__ == "__main__":
    # 开始训练数据
    bpnn = BPNN(train_x_data, train_y_data)
    bpnn_train = bpnn.train_gadient()

    train_y_data_tran = trans(train_y_data)

    predict_y_data_tran = trans(predict_y_data)

    # 数据多影响展示，随即挑选100条数据

    random_train_x_data = select(train_x_data, train_y_data_tran, 200)

    random_predict_x_data = select(predict_x_data, predict_y_data_tran, 100)

    train_output = trans(bpnn.predict(random_train_x_data[0]))

    predict_output = trans(bpnn.predict(random_predict_x_data[0]))

    figure(random_train_x_data[1], train_output, le='训练')

    figure(random_predict_x_data[1], predict_output, le='预测')

    costfig(bpnn_train[2])


__name__ == "__main__"




