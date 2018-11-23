# -*- coding：utf-8 -*-
# &Author  AnFany

# 引入数据
import pm25_RF_Data as data

# 引入模型
from sklearn.ensemble import  RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error as mse
import numpy as np

# 绘制不同参数下MSE的对比曲线
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt

# 根据K折交叉的结果确定比较好的参数组合，然后给出预测数据真实值和预测值的对比

# 对于回归而言，主要的参数就是随机森林中树的个数和特征的个数,其他参数均使用默认值

# 树的个数
trees = [50, 500, 1000, 2000, 4000, 7000]

# 随机选择的特征个数
tezheng = ['auto']  #  回归问题一般选用所有的特征

# 训练函数
def Train(data, treecount, tezh, yanzhgdata):
    model = RF(n_estimators=treecount, max_features=tezh)
    model.fit(data[:, :-1], data[:, -1])
    # 给出训练数据的预测值
    train_out = model.predict(data[:, :-1])
    # 计算MSE
    train_mse = mse(data[:, -1], train_out)

    # 给出验证数据的预测值
    add_yan = model.predict(yanzhgdata[:, :-1])
    # 计算MSE
    add_mse = mse(yanzhgdata[:, -1], add_yan)
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
                sumlist.append(xun + ya)
            sacelist['%s-%s' % (t, te)] = sumlist
            savedict['%s-%s' % (t, te)] = np.mean(np.array(sumlist))

    # 在结果字典中选择最小的
    zuixao = sorted(savedict.items(), key=lambda fu: fu[1])[0][0]
    # 然后再选出此方法中和值最小的折数
    xiao = sacelist[zuixao].index(min(sacelist[zuixao]))
    return zuixao, xiao, sacelist

# 根据字典绘制曲线
def duibi(exdict, you):
    plt.figure(figsize=(11, 7))
    for ii in exdict:
        plt.plot(list(range(len(exdict[ii]))), exdict[ii], \
                 label='%s%d折MSE均值:%.3f' % (ii, len(exdict[ii]), np.mean(np.array(exdict[ii]))), lw=2)
    plt.legend()
    plt.title('不同参数的组合MSE对比曲线[最优：%s]' % you)
    plt.savefig(r'C:\Users\GWT9\Desktop\method.jpg')
    return '不同方法对比完毕'

# 根据获得最有参数组合绘制真实和预测值的对比曲线
def recspre(exstr, predata, datadict, zhe):
    tree, te = exstr.split('-')
    model = RF(n_estimators=int(tree), max_features=te)
    model.fit(datadict[zhe]['train'][:, :-1], datadict[zhe]['train'][:, -1])

    # 预测
    yucede = model.predict(predata[:, :-1])
    # 为了便于展示，选100条数据进行展示
    zongleng = np.arange(len(yucede))
    randomnum = np.random.choice(zongleng, 100, replace=False)

    yucede_se = list(np.array(yucede)[randomnum])

    yuce_re = list(np.array(predata[:, -1])[randomnum])

    # 对比
    plt.figure(figsize=(17, 9))
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(yucede_se))), yucede_se, c='r', marker='*', label='预测', lw=2)
    plt.plot(list(range(len(yuce_re))), yuce_re, c='b', marker='.', label='真实', lw=2)
    plt.legend()
    plt.title('预测和真实值对比[最大树数%d]' % int(tree))

    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(yucede_se))), np.array(yuce_re) - np.array(yucede_se), 'k--', marker='s', label='真实-预测', lw=2)
    plt.legend()
    plt.title('预测和真实值相对误差')

    plt.savefig(r'C:\Users\GWT9\Desktop\duibi.jpg')
    return '预测真实对比完毕'

# 主函数

if __name__ == "__main__":
    zijian, zhehsu, xulie = Zuhe(data.dt_data)

    duibi(xulie, zijian)
    recspre(zijian, data.predict_data, data.dt_data, zhehsu)