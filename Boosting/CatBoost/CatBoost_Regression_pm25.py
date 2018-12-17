# -*- coding：utf-8 -*-
# &Author  AnFany

# 引入数据
import pm25_CatBoost_Data as data

# 引入模型
import catboost as cb
import numpy as np

# 绘制不同参数下MSE的对比曲线
from pylab import mpl
from collections import OrderedDict
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
from pylab import *
'''
模型阶段
'''

# 模型(树)的个数
models = [50000, 80000]

# 每个模型(树)的深度
cengs = [6, 7, 8]

# 视为类别型特征的字段的索引
label = [7]


# 训练函数
def Train(data, modelcount, censhu, yanzhgdata, predata, laidex=label):
    '''
    :param data: 训练数据
    :param modelcount: 模型的个数
    :param censhu: 每个模型的树的深度
    :param yanzhgdata: 用于验证的数据
    :param predata: 预测的数据
    :param laidex: 类别型特征的数据列号索引
    :return: 当前参数下该模型的训练、验证数据的结果
    '''

    # 用于回归的模型
    model = cb.CatBoostRegressor(iterations=modelcount, depth=censhu, learning_rate=0.8, loss_function='RMSE')

    #  开始训练数据
    model.fit(data[:, :-1], data[:, -1], cat_features=laidex, eval_set=(yanzhgdata[:, :-1], yanzhgdata[:, -1]))

    # 给出训练数据的评分
    train_mse = model.score(data[:, :-1], data[:, -1])

    # 计算预测数据的评分
    add_mse = model.score(yanzhgdata[:, :-1], yanzhgdata[:, -1])

    print(train_mse, add_mse)
    return train_mse, add_mse, model.predict(predata[:, :-1])

# 按照误差值从小到大排列的数据
def Pailie(realr, modelout, count=90):
    '''
    :param real: 预测数据集真实的数据
    :param modelout: 预测数据集的模型的输出值
    :param count: 进行数据对比的条数
    :return: 按照差值从小到大排列的数据
    '''
    relal_num = np.array(realr)
    modelout_num = np.array(modelout)
    # 随机选取
    fu = np.random.choice(list(range(len(realr))), count, replace=False)
    show_real, show_model = relal_num[fu], modelout_num[fu]
    # 计算差值
    sunnum = show_real - show_model
    # 首先组合三个数据列表为字典
    zuhedict = {ii: [show_real[ii], show_model[ii], sunnum[ii]] for ii in range(len(show_model))}
    # 字典按着值排序
    zhenshi = []
    yucede = []
    chazhi = []
    # 按着差值从大到小
    for jj in sorted(zuhedict.items(), key=lambda gy: gy[1][2]):
        zhenshi.append(jj[1][0])
        yucede.append(jj[1][1])
        chazhi.append(jj[1][2])
    return zhenshi, yucede, chazhi

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
    zuixao = min(savedict.items(), key=lambda fu: fu[1][1])[0]

    return zuixao, savedict, preresult[zuixao]


# 根据字典绘制不同参数下评分的对比柱状图
def duibi(exdict, you, kaudu=0.35):
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
    ax.set_ylabel('RMSE')
    ax.set_xlabel('不同的参数组合')
    # 设置刻度
    ax.set_xticks(ind)
    ax.set_xticklabels(palist)

    ax.legend()
    plt.title('不同参数的组合RMSE对比[最优：%s]' % you)
    plt.savefig(r'C:\Users\GWT9\Desktop\CatBoost_pm25.jpg')
    return '不同方法对比完毕'



# 绘制预测值，真实值，以及与两值的误差柱状图
def recspre(yzhenshide, yyucede):
    #  获得展示的数据
    yreal, ypre, cha = Pailie(yzhenshide['predict'][:, -1], yyucede)

    plt.figure(figsize=(18, 10))
    ax = plt.subplot(111)
    plt.grid()
    dign = np.arange(len(yreal))
    #  绘制真实值
    ax.scatter(dign, yreal, label='真实值', lw=2, color='blue', marker='*')
    #  绘制预测值
    ax.plot(dign, ypre, label='预测值', lw=2, color='red', linestyle='--', marker='.')
    #  绘制误差柱状图
    ax.bar(dign, cha, 0.1, label='真实值减去预测值', color='k')
    # 绘制0线
    ax.plot(dign, [0] * len(dign), lw=2, color='k')

    ax.set_ylim((int(min(cha)) - 1, int(max([max(yreal), max(ypre)]))))
    ax.set_xlim((0, len(dign)))

    ax.legend(loc='upper center')
    ax.set_title('北京市Pm2.5预测数据集结果对比')
    plt.savefig(r'C:\Users\GWT9\Desktop\CatBoost_duibi.jpg')

    return '完毕'


# 最终的主函数
    
if __name__ == "__main__":
    zijian, sdict, yudd = Zuhe(data.data_dict)
    # 绘制图像
    duibi(sdict, zijian)
    recspre(data.data_dict, yudd)
