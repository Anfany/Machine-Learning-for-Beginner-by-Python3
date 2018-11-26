# -*- coding：utf-8 -*-
# &Author  AnFany


# 引入数据
import adult_RF_Data as data

# 引入随机森林分类模型，支持多类别
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np

# 格式化输出混淆矩阵
from prettytable import PrettyTable as PT

# 绘制不同参数下F1度量的对比曲线
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt

# 根据K折交叉的结果确定比较好的参数组合，然后给出预测数据真实值和预测值的对比

# 对于回归而言，主要的参数就是随机森林中树的个数和特征的个数,其他参数均使用默认值

# 树的个数
trees = [100, 500, 1000, 2000]

# 随机选择的特征个数
tezheng = ['sqrt']  #  分类问题一般选用平方根个数的特征

# 混淆矩阵的函数
def Tom(reallist, prelist):
    '''
    :param reallist: 真实的类别列表
    :param prelist:  预测的类别列表
    :return: 每个类别预测为所有类别的个数字典
    '''
    coundict = {}
    for jj in list(set(reallist)):
        coundict[jj] = {}
        for hh in list(set(reallist)):
            coundict[jj][hh] = len([i for i, j in zip(reallist, prelist) if i == jj and j == hh])
    return coundict

# 定义输出混淆矩阵的函数
def ConfuseMatrix(reallist, prelist):
    '''
    :param reallist: 真实的类别列表
    :param prelist: 预测的类别列表
    :return: 输出混淆矩阵
    '''
    zidian = Tom(reallist, prelist)
    lieming = sorted(zidian.keys())
    table = PT(['混淆矩阵'] + ['预测%s'% d for d in lieming])
    for jj in lieming:
        table.add_row(['实际%s' % jj] + [zidian[jj][kk] for kk in lieming])
    return table

#  计算F1度量的函数
def fmse(realist, prelist):  # 对于多类别每个类都要计算召回率
    '''
    :param realist: 真实的类别列表
    :param prelist: 预测的类别列表
    :return: F1度量
    '''
    condict = Tom(realist, prelist)
    zongshu = 0
    zhengque = 0
    zhao_cu = []  # 存储每个类别的召回率
    for cu in condict:
        zq = 0
        zs = 0
        for hh in condict[cu]:
            geshu = condict[cu][hh]
            if cu == hh:
                zhengque += geshu
                zq = geshu
            zongshu += geshu
            zs += geshu
        zhao_cu.append(zq / zs)
    # 计算精确率
    jingque = zhengque / zongshu
    # 计算类别召回率
    zhaohui = np.mean(np.array(zhao_cu))
    # f1度量
    f_degree = 2 * jingque * zhaohui / (jingque + zhaohui)
    return f_degree, jingque, zhaohui


# 训练函数
def Train(data, treecount, tezh, yanzhgdata):
    model = RFC(n_estimators=treecount, max_features=tezh, class_weight='balanced')
    model.fit(data[:, :-1], data[:, -1])
    # 给出训练数据的预测值
    train_out = model.predict(data[:, :-1])
    # 计算MSE
    train_mse = fmse(data[:, -1], train_out)[0]

    # 给出验证数据的预测值
    add_yan = model.predict(yanzhgdata[:, :-1])
    # 计算f1度量
    add_mse = fmse(yanzhgdata[:, -1], add_yan)[0]
    print(train_mse, add_mse)
    return train_mse, add_mse

# 最终确定组合的函数
def Zuhe(datadict, tre=trees, tezhen=tezheng):
    # 存储结果的字典
    savedict = {}
    # 存储序列的字典
    sacelist = {}
    for t in tre:
        for te in tezhen:
            print(t, te)
            sumlist = []
            # 因为要展示折数，因此要按序开始
            ordelist = sorted(list(datadict.keys()))
            for jj in ordelist:
                xun, ya = Train(datadict[jj]['train'], t, te, datadict[jj]['test'])
                sumlist.append((xun + ya) / 2)
            sacelist['%s-%s' % (t, te)] = sumlist
            savedict['%s-%s' % (t, te)] = np.mean(np.array(sumlist))

    # 在结果字典中选择最大的
    zuixao = sorted(savedict.items(), key=lambda fu: fu[1], reverse=True)[0][0]
    # 然后再选出此方法中和值最大的折数
    xiao = sacelist[zuixao].index(max(sacelist[zuixao]))
    return zuixao, xiao, sacelist

# 根据字典绘制曲线
def duibi(exdict, you):
    plt.figure(figsize=(11, 7))
    for ii in exdict:
        plt.plot(list(range(len(exdict[ii]))), exdict[ii], \
                 label='%s%d折F1均值:%.3f' % (ii, len(exdict[ii]), np.mean(np.array(exdict[ii]))), lw=2)
    plt.legend()
    plt.title('不同参数的组合F1对比曲线[最优：%s]' % you)
    plt.savefig(r'C:\Users\GWT9\Desktop\method_adult.jpg')
    return '不同方法对比完毕'

# 根据获得最有参数组合绘制真实和预测值的对比曲线
def recspre(exstr, predata, datadict, zhe):
    tree, te = exstr.split('-')
    model = RFC(n_estimators=int(tree), max_features=te)

    model.fit(datadict[zhe]['train'][:, :-1], datadict[zhe]['train'][:, -1])

    # 预测
    yucede = model.predict(predata[:, :-1])
    # 计算混淆矩阵

    print(ConfuseMatrix(predata[:, -1], yucede))

    return fmse(predata[:, -1], yucede)

# 主函数

if __name__ == "__main__":
    zijian, zhehsu, xulie = Zuhe(data.dt_data)
    # 绘制方法组合的对比曲线
    duibi(xulie, zijian)
    # 计算预测数据的f1度量，精确率以及召回率
    f1, jing, zhao = recspre(zijian, data.predict_data, data.dt_data, zhehsu)
    print('F1度量：{}, 精确率：{}, 召回率：{}'.format(f1, jing, zhao))


