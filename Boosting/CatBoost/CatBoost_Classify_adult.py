# -*- coding：utf-8 -*-
# &Author  AnFany


# 引入数据
import adult_CatBoost_Data as data

# 引入模型
import catboost as cb
import numpy as np
from collections import OrderedDict
# 格式化输出混淆矩阵
from prettytable import PrettyTable as PT

# 绘制不同参数下F1度量的对比曲线
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt


# 主要的参数就是模型的个数以及单个树的深度

# 模型的个数，也就是树的个数
cengs = [6, 7, 8]

# 单个树的深度
models = [100, 200, 300]

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

# 定义输出混淆矩阵的函数,需要将类别值和原始的进行转换
def ConfuseMatrix(reallist, prelist, dcix=data.exdixxt):
    '''
    :param reallist: 真实的类别列表
    :param prelist: 预测的类别列表
    :return: 输出混淆矩阵
    '''

    # 首先将字典的键值互换
    ruid = {}
    for jj in dcix:
        ruid[dcix[jj]] = jj

    zidian = Tom(reallist, prelist)
    lieming = sorted(zidian.keys())
    table = PT(['混淆矩阵'] + ['预测%s' % ruid[d] for d in lieming])
    for jj in lieming:
        table.add_row(['实际%s' % ruid[jj]] + [zidian[jj][kk] for kk in lieming])
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
def Train(data, modelcount, censhu, yanzhgdata, predata, cat=data.catind):
    model = cb.CatBoostClassifier(iterations=modelcount, depth=censhu, learning_rate=0.5, loss_function='Logloss',
                                  logging_level='Verbose')

    model.fit(data[:, :-1], data[:, -1], cat_features=cat, eval_set=(yanzhgdata[:, :-1], yanzhgdata[:, -1]))
    # 给出训练数据的分数
    train_mse = model.score(data[:, :-1], data[:, -1])

    # 给出验证数据的分数
    add_mse = model.score(yanzhgdata[:, :-1], yanzhgdata[:, -1])
    print(train_mse, add_mse)
    # 输出预测数据的预测值
    preout = model.predict(predata[:, :-1])
    return train_mse, add_mse, preout

# 最终确定组合的函数
def Zuhe(datadict, tre=models, tezhen=cengs):
    '''
    :param datadict: 存储数据集的字典
    :param tre: 模型树的棵数
    :param tezhen: 每棵树的深度
    :return: 最佳的参数组合，以及个参数组合对应的训练、验证数据的评分字典
    '''

    # 存储结果的字典，按序的。
    savedict = OrderedDict()

    # 存储每一次预测的结果字典
    preresult = {}

    for t in tre:
        for te in tezhen:
            print(t, te)
            # 训练
            xun, ya, psult = Train(datadict['train'], t, te, datadict['test'], datadict['predict'])

            # 存储结果
            savedict['%s-%s' % (t, te)] = [xun, ya]

            preresult['%s-%s' % (t, te)] = psult

    # 在结果字典中选择验证数据的评分最小的参数组合
    zuixao = max(savedict.items(), key=lambda fu: fu[1][1])[0]

    return zuixao, savedict, preresult[zuixao]


# 根据字典绘制不同参数下评分的对比柱状图
def duibi(exdict, you, kaudu=0.3):
    '''
    :param exdict: 不同参数组合下的训练、测试数据的评分
    :param you: 最优的参数组合
    :return: 直方图
    '''
    # 参数组合列表
    palist = exdict.keys()
    # 对应的训练数据的评分
    trsore = [exdict[hh][0] for hh in palist]
    # 对应的测试数据的评分
    tesore = [exdict[hh][1] for hh in palist]

    # 开始绘制柱状图
    fig, ax = plt.subplots()
    # 柱的个数
    ind = np.array(list(range(len(trsore))))
    # 绘制柱状
    ax.bar(ind - kaudu, trsore, kaudu, color='SkyBlue', label='训练')
    ax.bar(ind, tesore, kaudu, color='IndianRed', label='测试')
    # xy轴的标签
    ax.set_ylabel('分数')
    ax.set_xlabel('不同的参数组合')
    # 设置刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(palist)

    ax.set_ylim((0.8, 0.9))

    ax.grid()

    ax.legend()
    plt.title('不同参数的组合RMSE对比[最优：%s]' % you)
    plt.savefig(r'C:\Users\GWT9\Desktop\CatBoost_adult.jpg')
    return '不同方法对比完毕'



# 输出预测数据的混淆矩阵以及各种指标
def recspre(yzhenshide, yyucede):
    # 计算混淆矩阵
    print(ConfuseMatrix(yzhenshide['predict'][:, -1], yyucede))
    return fmse(yzhenshide['predict'][:, -1], yyucede)

# 主函数

if __name__ == "__main__":
    zijian, fend, yunum = Zuhe(data.datt)
    # 绘制方法组合的对比曲线
    duibi(fend, zijian)
    # 计算预测数据的f1度量，精确率以及召回率
    f1, jing, zhao = recspre(data.datt, yunum)
    print('F1度量：{}, 精确率：{}, 召回率：{}'.format(f1, jing, zhao))