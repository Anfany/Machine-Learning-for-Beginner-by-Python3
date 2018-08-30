# -*- coding：utf-8 -*-
# &Author AnFany

# SMO算法实现支持向量机核函数二分类

"""
第一部分：引入库
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 中文字体名称
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

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
    def __init__(self, polyd=3, rbfsigma=0.22, tanhbeta=0.6, tanhtheta=-0.6):
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
            return np.tanh(self.tanhbeta * ((sx * y).sum(axis=2).T) + self.tanhtheta)

    # 多项式核函数
    def poly(self, x, y):  # 输出的结果shape=(len(y), len(x))
        x, y = self.trans(x), self.trans(y)
        if len(x) == 1:
            return (x * y).sum(axis=1, keepdims=True) ** self.polyd
        else:
            sx = x.reshape(x.shape[0], -1, x.shape[1])
            return (sx * y).sum(axis=2).T ** self.polyd

    #  高斯核函数
    def rbf(self, x, y):  # 输出的结果shape=(len(y), len(x))
        x, y = self.trans(x), self.trans(y)
        if len(x) == 1 and len(y) == 1:
            return np.exp(self.linear((x - y), (x - y)) / (-2 * self.rbfsigma ** 2))
        elif len(x) == 1 and len(y) != 1:
            return np.exp((np.power(x - y, 2)).sum(axis=1, keepdims=True) / (-2 * self.rbfsigma ** 2))
        else:
            sx = x.reshape(x.shape[0], -1, x.shape[1])
            return np.exp((np.power(sx - y, 2)).sum(axis=2).T / (-2 * self.rbfsigma ** 2))


#  构建SVM的结构
class SVM:
    def __init__(self, feature, labels, kernel='rbf', C=0.8, toler=0.001, times=300):
        #  训练样本的属性数据、标签数据
        self.feature = feature
        self.labels = labels


        # SMO算法变量
        self.C = C
        self.toler = toler
        self.alphas = np.zeros(len(self.feature))
        self.b = 0
        self.eps = 0.0001  # 选择拉格朗日因子

        # 核函数
        self.kernel = eval('KERNEL().' + kernel)

        # 拉格朗日误差序列
        self.errors = [self.get_error(i) for i in range(len(self.feature))]

        #  循环的最大次数
        self.times = times


    # 计算分割线的值
    def line_num(self, x):
        ks = self.kernel(x, self.feature)
        wx = np.matrix(self.alphas * self.labels) * ks
        num = wx + self.b
        return num[0][0]

    #  获得编号为i的样本对应的误差
    def get_error(self, i):
        x, y = self.feature[i], self.labels[i]
        error = self.line_num(x) - y
        return error

    #  更改拉格朗日因子后，更新所有样本对应的误差
    def update_errors(self):
        self.errors = [self.get_error(i) for i in range(len(self.feature))]

    # 判断是否违背KKT条件
    def meet_kkt(self, i):
        alpha, x = self.alphas[i], self.feature[i]
        if alpha == 0:
            return self.line_num(x) >= 1
        elif alpha == self.C:
            return self.line_num(x) <= 1
        else:
            return self.line_num(x) == 1

"""
第三部分：构建SMO算法需要的函数
"""
#  alpha的值到L和H之间.
def clip(alpha, L, H):
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha

#  随机选择一个和当前因子不同的因子
def select_j_rand(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i + 1:]
    return np.random.choice(seq)

#  启发式选择第二个因子
def select_j(i, svm):
    errors = svm.errors
    valid_indices = [i for i, a in enumerate(svm.alphas) if 0 < a < svm.C]
    if len(valid_indices) > 1:
        j = -1
        max_delta = 0
        for k in valid_indices:
            if k == i:
                continue
            delta = abs(errors[i] - errors[j])
            if delta > max_delta:
                j = k
                max_delta = delta
    else:
        j = select_j_rand(i, len(svm.feature))
    return j

#  优化已经选择的一对因子
def take_step(i, j, svm):
    #  首先获得最新的误差列表
    svm.update_errors()

    #  拉格朗日因子及其对应的样本数据，标签数据，误差
    a_i, x_i, y_i, e_i = svm.alphas[i], svm.feature[i], svm.labels[i], svm.errors[i]
    a_j, x_j, y_j, e_j = svm.alphas[j], svm.feature[j], svm.labels[j], svm.errors[j]

    # 计算单样本之间的核函数
    k_ii, k_jj, k_ij = svm.kernel(x_i, x_i), svm.kernel(x_j, x_j), svm.kernel(x_i, x_j)

    eta = k_ii + k_jj - 2 * k_ij

    if eta <= 0:
        return 0

    a_i_old, a_j_old = a_i, a_j
    a_j_new = a_j_old + y_j * (e_i - e_j) / eta

    # 对alpha进行修剪
    if y_i != y_j:
        Lmax = max(0, a_j_old - a_i_old)
        Hmin = min(svm.C, svm.C + a_j_old - a_i_old)
    else:
        Lmax = max(0, a_i_old + a_j_old - svm.C)
        Hmin = min(svm.C, a_j_old + a_i_old)

    a_j_new = clip(a_j_new, Lmax, Hmin)
    a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)

    if abs(a_j_new - a_j_old) < svm.eps:
        return 0

    #  更新拉格朗日因子
    svm.alphas[i], svm.alphas[j] = a_i_new, a_j_new
    #  更新误差
    svm.update_errors()

    # 更新阈值b
    b_i = -e_i - y_i * k_ii * (a_i_new - a_i_old) - y_j * k_ij * (a_j_new - a_j_old) + svm.b
    b_j = -e_j - y_i * k_ij * (a_i_new - a_i_old) - y_j * k_jj * (a_j_new - a_j_old) + svm.b

    if 0 < a_i_new < svm.C:
        bnum = b_i
    elif 0 < a_j_new < svm.C:
        bnum = b_j
    else:
        bnum = (b_i + b_j) / 2

    # 更新b值
    svm.b = bnum

    return 1


# 给定第一个alpha因子， 检测对应alpha是否符合KKT条件并选取第二个alpha进行迭代.
def examine_example(i, svm):

    e_i, y_i, alpha = svm.errors[i], svm.labels[i], svm.alphas[i]
    r = e_i * y_i

    # 是否违反KKT条件
    if (r < -svm.toler and alpha < svm.C) or (r > svm.toler and alpha > 0):
        #  启发式选择
        j = select_j(i, svm)
        return take_step(i, j, svm)
    else:
        return 0


# Platt SMO算法实现
def platt_smo(svm):
    # 循环次数
    it = 0
    # 遍历所有alpha的标记
    entire = True
    pair_changed = 0
    while it < svm.times:  # and (pair_changed > 0 or entire):
        pair_changed = 0
        if entire:
            for i in range(len(svm.feature)):
                pair_changed += examine_example(i, svm)
                print('全部样本 改变的因子对数: %s' % pair_changed)
        else:
            alphas = svm.alphas
            non_bound_indices = [i for i in range(len(svm.feature)) if alphas[i] > 0 and alphas[i] < svm.C]
            for i in non_bound_indices:
                pair_changed += examine_example(i, svm)
                print('非边界 改变的因子数：%s' %pair_changed)
        #  循环次数
        it += 1

        #  更改边界
        if entire:
            entire = False
        elif pair_changed == 0:
            entire = True

        print('外层循环的次数: %s' % it)

    return svm.alphas, svm.b


#  预测函数
def predict(svm, prefeature):
    prlt = np.array((np.array([svm.alphas]).reshape(-1, 1) * svm.kernel(prefeature, svm.feature) * np.array([svm.labels]).reshape(-1, 1)).sum(axis=0) + svm.b)
    signre = np.sign(prlt[0])
    return signre

#  获得正确率函数
def getacc(svm, prefeature, prelabel):
    predu = predict(svm, prefeature)
    # 计算正确率
    sub = np.array(predu - prelabel)
    acc = len(sub == 0) / len(prelabel)
    return acc


# 引入心脏病数据
import SVM_Classify_Data as sdata


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
        resu = SVM(feature=xd, labels=yd, kernel=he)
        # 开始训练
        platt_smo(resu)
        # 训练完，储存训练、测试的精确度结果
        trainacc.append(getacc(resu, xd, yd))
        testacc.append(getacc(resu, texd, teyd))
        # 保存支持向量的个数
        count = len(resu.alphas < 0.001)
        vec.append(count)
        sign.append(jj)

    # 绘制多y轴图
    fig, host = plt.subplots()
    # 用来控制多y轴
    par1 = host.twinx()
    #  多条曲线
    p1, = host.plot(sign, trainacc, "b-", marker='8', label='训练', linewidth=2)
    pp, = host.plot(sign, testacc, "b--", marker='*', label='测试', linewidth=2)
    p2, = par1.plot(sign, vec, "r-", marker='8', label='支持向量个数', linewidth=2)
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



