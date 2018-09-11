# -*- coding：utf-8 -*-
# &Author AnFany

# SMO算法实现支持向量机回归

"""
第一部分：引入库
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# 引入数据
import SVM_Regression_Data as rdata

"""
第二部分：构建核函数以及SVM的结构
"""

#  构建核函数
class KERNEL:
    """
    linear：线性   rbf：高斯  sigmoid：Sigmoid型  poly：多项式
    核函数：注意输入数据的shape以及输出数据的shape。
    xVSy包括3种情况：单样本VS单样本  单样本VS多样本  多样本VS多样本
    """

    def __init__(self, polyd=3, rbfsigma=0.2, tanhbeta=0.6, tanhtheta=-0.6):
        self.polyd = polyd
        self.rbfsigma = rbfsigma
        self.tanhbeta = tanhbeta
        self.tanhtheta = tanhtheta

    def trans(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = np.array([x])
        return x

    # 线性核函数
    def linear(self, x, y):  # 输出的结果shape=(len(y), len(x))
        x, y = self.trans(x), self.trans(y)
        if len(x) == 1:
            return (x * y).sum(axis=1, keepdims=True)
        else:
            sx = x.reshape(x.shape[0], -1, x.shape[1])
            return (sx * y).sum(axis=2).T

    # Singmoid型核函数
    def sigmoid(self, x, y):  # 输出的结果shape=(len(y), len(x))
        x, y = self.trans(x), self.trans(y)
        if len(x) == 1:
            return np.tanh(self.tanhbeta * ((x * y).sum(axis=1, keepdims=True)) + self.tanhtheta)
        else:
            sx = x.reshape(x.shape[0], -1, x.shape[1])
            return np.tanh(self.tanhbeta * (sx * y).sum(axis=2).T + self.tanhtheta)

    # 多项式核函数
    def poly(self, x, y):  # 输出的结果shape=(len(y), len(x))
        x, y = self.trans(x), self.trans(y)
        if len(x) == 1:
            return (x * y).sum(axis=1, keepdims=True) ** self.polyd
        else:
            sx = x.reshape(x.shape[0], -1, x.shape[1])
            return (sx * y).sum(axis=2).T ** self.polyd

    # 高斯核函数
    def rbf(self, x, y):  # 输出的结果shape=(len(y), len(x))
        x, y = self.trans(x), self.trans(y)
        if len(x) == 1 and len(y) == 1:
            return np.exp(self.linear((x - y), (x - y)) / (-2 * self.rbfsigma ** 2))
        elif len(x) == 1 and len(y) != 1:
            return np.exp((np.power(x - y, 2)).sum(axis=1, keepdims=True) / (-2 * self.rbfsigma ** 2))
        else:
            sx = x.reshape(x.shape[0], -1, x.shape[1])
            return np.exp((np.power(sx - y, 2)).sum(axis=2).T / (-2 * self.rbfsigma ** 2))


# 构建SVM的结构
class SVR:
    def __init__(self, feature, labels, kernel='rbf', C=0.8, toler=0.001, epsilon=0.001, times=100, eps=0.0001):
        #  训练样本的属性数据、标签数据
        self.feature = feature
        self.labels = labels

        # SMO算法变量
        self.C = C
        self.toler = toler

        self.alphas = np.zeros(len(self.feature))
        self.alphas_star = np.zeros(len(self.feature))

        self.b = 0
        self.eps = eps  # 选择拉格朗日因子
        self.epsilon = epsilon

        # 核函数
        self.kernel = eval('KERNEL().' + kernel)

        # 拉格朗日误差序列
        self.errors = [self.get_error(i) for i in range(len(self.feature))]

        #  循环的最大次数
        self.times = times

    # 计算分割线的值,x为单样本
    def line_num(self, x):
        ks = self.kernel(x, self.feature)
        wx = np.matrix(self.alphas - self.alphas_star) * ks
        num = np.array(wx + self.b)
        return num[0][0]

    #  获得编号为i的样本对应的误差
    def get_error(self, i):
        x, y = self.feature[i], self.labels[i]
        error = self.line_num(x) - y
        return error

    #  更改拉格朗日因子后，更新所有样本对应的误差
    def update_errors(self):
        self.errors = [self.get_error(i) for i in range(len(self.feature))]


    # 预测函数
    def predictall(self, prefeat):
        ks = self.kernel(prefeat, self.feature)
        wx = np.matrix(self.alphas - self.alphas_star) * ks
        num = np.array(wx + self.b)
        return num[0]


"""
第三部分：构建SMO算法需要的函数
"""


def takestep(svr, i1, i2):
    if i1 == i2:
        return 0
    alpha1, alphas1 = svr.alphas[i1], svr.alphas_star[i1]

    alpha2, alphas2 = svr.alphas[i2], svr.alphas_star[i2]

    phi1, phi2 = svr.errors[i1], svr.errors[i2]

    x1, x2 = svr.feature[i1], svr.feature[i2]

    y1, y2 = svr.labels[i1], svr.labels[i2]

    k11, k12, k22 = svr.kernel(x1, x1)[0][0], svr.kernel(x1, x2)[0][0], svr.kernel(x2, x2)[0][0]

    eta = 2 * k12 - k11 - k22

    gamma = alpha1 - alphas1 + alpha2 - alphas2

    case1 = case2 = case3 = case4 = finished = 0

    a1old, a1olds = alpha1, alphas1

    delta_phi = phi1 - phi2

    while not finished:
        if (case1 == 0) and (alpha1 > 0 or (alphas1 == 0 and delta_phi > 0)) \
                and (alpha2 > 0 or (alphas2 == 0 and delta_phi < 0)):
            # 计算L和H (a1, a2)
            L = max(0, gamma - svr.C)
            H = min(svr.C, gamma)
            if L < H:
                if eta > 0:
                    a2 = alpha2 - delta_phi / eta
                    a2 = min(a2, H)
                    a2 = max(L, a2)
                    a1 = alpha1 - a2 + alpha2
                else:
                    a2 = L
                    a1 = alpha1 - (a2 - alpha2)
                    object1 = -0.5 * a1 * a1 * eta + a1 * (delta_phi + (a1old - a1olds) * eta)
                    a2 = H
                    a1 = alpha1 - (a2 - alpha2)
                    object2 = -0.5 * a1 * a1 * eta + a1 * (delta_phi + (a1old - a1olds) * eta)
                    if object1 > object2:
                        a2 = L
                    else:
                        a2 = H
                    a1 = alpha1 - (a2 - alpha2)
                if abs(delta_phi - eta * (a1 - alpha1)) > svr.epsilon:
                    svr.alphas[i1] = a1
                    svr.alphas[i2] = a2
            else:
                finished = 1
            case1 = 1

        elif (case2 == 0) and (alpha1 > 0 or (alphas1 == 0 and delta_phi > 2 * svr.epsilon)) \
                and (alphas2 > 0 or (alpha2 == 0 and delta_phi > 2 * svr.epsilon)):
            # 计算L和H (a1, a2*)
            L = max(0, gamma)
            H = min(svr.C, svr.C + gamma)

            if L < H:
                if eta > 0:
                    a2 = alphas2 + (delta_phi - 2 * svr.epsilon) / eta
                    a2 = min(a2, H)
                    a2 = max(L, a2)
                    a1 = alpha1 + a2 - alphas2
                else:
                    a2 = L
                    a1 = alpha1 + (a2 - alphas2)
                    object1 = -0.5 * a1 * a1 * eta - 2 * svr.epsilon * a1 + a1 * (delta_phi + (a1old - a1olds) * eta)
                    a2 = H
                    a1 = alpha1 + (a2 - alpha2)
                    object2 = -0.5 * a1 * a1 * eta - 2 * svr.epsilon * a1 + a1 * (delta_phi + (a1old - a1olds) * eta)
                    if object1 > object2:
                        a2 = L
                    else:
                        a2 = H
                    a1 = alpha1 + (a2 - alphas2)

                # 判断变化大小
                if abs(delta_phi - eta * (a1 - alpha1)) > svr.epsilon:
                    svr.alphas[i1] = a1
                    svr.alphas_star[i2] = a2
            else:
                finished = 1
            case2 = 1


        elif (case3 == 0) and (alphas1 > 0 or (alpha1 == 0 and delta_phi < 2 * svr.epsilon)) \
                and (alpha2 > 0 or (alphas2 == 0 and delta_phi < 2 * svr.epsilon)):
            # 计算L和H (a1*, a2)
            L = max(0, -gamma)
            H = min(svr.C, svr.C - gamma)

            if L < H:
                if eta > 0:
                    a2 = alpha2 - (delta_phi - 2 * svr.epsilon) / eta
                    a2 = min(a2, H)
                    a2 = max(L, a2)
                    a1 = alphas1 + a2 - alpha2
                else:
                    a2 = L
                    a1 = alphas1 + (a2 - alpha2)
                    object1 = -0.5 * a1 * a1 * eta - 2 * svr.epsilon * a1 - a1 * (delta_phi + (a1old - a1olds) * eta)
                    a2 = H
                    a1 = alphas1 + (a2 - alpha2)
                    object2 = -0.5 * a1 * a1 * eta - 2 * svr.epsilon * a1 - a1 * (delta_phi + (a1old - a1olds) * eta)
                    if object1 > object2:
                        a2 = L
                    else:
                        a2 = H
                    a1 = alphas1 + (a2 - alpha2)

                # 判断变化大小
                if abs((delta_phi - eta * (-a1 + alphas1))) > svr.epsilon:
                    svr.alphas_star[i1] = a1
                    svr.alphas[i2] = a2
            else:
                finished = 1
            case3 = 1


        elif (case4 == 0) and (alphas1 > 0 or (alpha1 == 0 and delta_phi < 0)) \
                and (alphas2 > 0 or (alpha2 == 0 and delta_phi > 0)):
            # 计算L和H (a1*, a2*)
            L = max(0, -gamma - svr.C)
            H = min(svr.C, - gamma)

            if L < H:
                if eta > 0:
                    a2 = alphas2 + delta_phi / eta
                    a2 = min(a2, H)
                    a2 = max(L, a2)
                    a1 = alphas1 - a2 + alphas2
                else:
                    a2 = L
                    a1 = alphas1 - (a2 - alphas2)
                    object1 = -0.5 * a1 * a1 * eta - a1 * (delta_phi + (a1old - a1olds) * eta)
                    a2 = H
                    a1 = alphas1 - (a2 - alphas2)
                    object2 = -0.5 * a1 * a1 * eta - a1 * (delta_phi + (a1old - a1olds) * eta)
                    if object1 > object2:
                        a2 = L
                    else:
                        a2 = H
                    a1 = alphas1 - (a2 - alphas2)
                    # 判断变化大小
                if abs((delta_phi - eta * (-a1 + alphas1))) > svr.epsilon:
                    svr.alphas_star[i1] = a1
                    svr.alphas_star[i2] = a2
            else:
                finished = 1
            case4 = 1
        else:
            finished = 1

        delta_phi = delta_phi - eta * ((alpha1 - alphas1) - (a1old - a1olds))


    # 更新b值
    b1, b2 = 0, 0
    for ii in range(len(svr.feature)):
        b1 += (svr.alphas[ii] - svr.alphas_star[ii]) * svr.kernel(svr.feature[ii], x1)[0][0]
        b2 += (svr.alphas[ii] - svr.alphas_star[ii]) * svr.kernel(svr.feature[ii], x2)[0][0]

    b1 = y1 - b1
    b2 = y2 - b2

    b12 = (b1 + b2) / 2
    svr.b = b12


    # 更新误差
    svr.update_errors()

    if abs(delta_phi) > svr.epsilon:
        return 1
    else:
        return 0

def examineExample(svr, i2):
    alpha2 = svr.alphas[i2]
    alphas2 = svr.alphas_star[i2]
    phi2 = svr.errors[i2]

    if (phi2 > svr.epsilon and alphas2 < svr.C) or (phi2 < svr.epsilon and alphas2 > 0) \
            or(-phi2 > svr.epsilon and alpha2 < svr.C) or (-phi2 > svr.epsilon and alpha2 > 0):
        manzu = [io for io in range(len(svr.feature)) if svr.alphas[io] != 0 and svr.alphas[io] != svr.C]
        if len(manzu) > 1:
            delta_phi = 0
            i1 = 0
            for hi in range(len(svr.feature)):
                # 选择最大的启发式选择
                yphi = svr.errors[hi]
                if abs(yphi - phi2) > delta_phi:
                    delta_phi = abs(yphi - phi2)
                    i1 = hi
            if takestep(svr, i1, i2):
                return 1
        # 随机边界样本选择
        for jj in range(len(svr.feature)):
            i1 = np.random.choice(np.arange(len(svr.feature)), 1)[0]
            if svr.alphas[i1] != 0 and svr.alphas[i1] != svr.C:
                if takestep(svr, i1, i2):
                    return 1
        # 随机全部样本选择
        for jj in range(len(svr.feature)):
            i1 = np.random.choice(np.arange(len(svr.feature)), 1)[0]
            if takestep(svr, i1, i2):
                return 1
    return 0



# 主要函数
def mainfun(svr):
    numChanged = 0
    examineAll = 1
    SigFig = -100
    Loopcounter = 0

    while ((numChanged > 0 or examineAll) or SigFig < 3) and Loopcounter < 10000:
        Loopcounter += 1
        numChanged = 0
        if examineAll:
            print('全部样本')
            for ig in range(len(svr.feature)):
                numChanged += examineExample(svr, ig)
        else:
            print('边界样本')
            manzu = [ipo for ipo in range(len(svr.feature)) if svr.alphas[ipo] != 0 and svr.alphas[ipo] != svr.C]
            for gi in manzu:
                numChanged += examineExample(svr, gi)

        if Loopcounter % 2 == 0:
            MinimumNumChanged = max(1, 0.1 * len(svr.alphas))
        else:
            MinimumNumChanged = 1

        if examineAll == 1:
            examineAll = 0
        else:
            if numChanged < MinimumNumChanged:
                examineAll = 1
        dobject = 0
        pobject = 0

        for ghi in range(len(svr.feature)):
            p1 = svr.feature[ghi]
            dobject += max(0, svr.line_num(p1) - svr.labels[ghi] - svr.epsilon) * (svr.C - svr.alphas_star[ghi])
            dobject -= min(0, svr.line_num(p1) - svr.labels[ghi] - svr.epsilon) * svr.alphas_star[ghi]
            dobject += min(0, svr.labels[ghi] - svr.epsilon - svr.line_num(p1)) * (svr.C - svr.alphas[ghi])
            dobject -= max(0, svr.labels[ghi] - svr.epsilon - svr.line_num(p1)) * svr.alphas[ghi]


            p1 = svr.feature[ghi]
            pobject += (0.5 * (svr.alphas[ghi] - svr.alphas_star[ghi]) * (svr.line_num(p1) - svr.b))
            pobject -= (svr.epsilon * (svr.alphas[ghi] + svr.alphas_star[ghi]))
            pobject += (svr.labels[ghi] * (svr.alphas[ghi] - svr.alphas_star[ghi]))

        print('gggggggggggggggggggggggggggggggggggg', pobject, dobject)
        pobject += dobject

        SigFig = np.log10(dobject / (abs(pobject) + 1))
        print('SigFig = %.9f' % SigFig)

    print('结束训练')
    return svr.alphas, svr.alphas_star, svr.b


# 返回训练数据、预测数据的输出值
def pl(trfe, trla, prfe, maxnum, minnum):
    # 建立结构
    svre = SVR(feature=trfe, labels=trla)
    mainfun(svre)
    traiout = svre.predictall(trfe)
    preout = svre.predictall(prfe)

    # 首先是数据转化范围
    traiout = traiout * (maxnum - minnum) + minnum
    preout = preout * (maxnum - minnum) + minnum

    return traiout, preout


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
    outtri, poupre = pl(datasvr[0], datasvr[1].T[0], datasvr[2], datasvr[4][0], datasvr[4][1])

    trii = datasvr[1].T[0] * (datasvr[4][0] - datasvr[4][1]) + datasvr[4][1]
    huitu(outtri, trii, c=['b', 'k'], sign='训练', cudu=3)

    prii = datasvr[3].T[0] * (datasvr[4][0] - datasvr[4][1]) + datasvr[4][1]
    huitu(poupre, prii, c=['b', 'k'], sign='预测', cudu=3)
