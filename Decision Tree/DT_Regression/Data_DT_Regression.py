# -*- coding：utf-8 -*-
# &Author  AnFany


import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\GWT9\Desktop\PRSA_data_2010.1.1-2014.12.31.csv')

#  因为决策树回归和决策树分类程序，相似性很大，唯一的不同在于分类计算基尼系数，而回归计算MSE。
#  因此回归的数据结构在此处和分类的相似

#  因为原始数据中有些变量的值是离散的，但是是整数表示，因此这里需要将其变为字符串，
#  这主要涉及的变量包括：year	month	day	hour

'''第一部分：缺失值的处理'''
#  因为Pm2.5是目标数据，对于缺失值直接删除这一条记录

# 定义删除目标值为空值的行的函数, 其他列为控制则自动填充的函数,并将目标变量放置在数据集最后一列
def DeleteTargetNan(exdata, targetstr):
    #  首先判断目标字段是否有缺失值
    if exdata[targetstr].isnull().any():
        #  首先确定缺失值的行数
        loc = exdata[targetstr][data[targetstr].isnull().values == True].index.tolist()
        #  然后删除这些行
        exdata = exdata.drop(loc)
    # 凡是有缺失值的再一起利用此行的均值填充
    exdata = exdata.fillna(exdata.mean())
    # 将目标字段至放在最后的一列
    targetnum = exdata[targetstr].copy()
    del exdata[targetstr]
    exdata[targetstr] = targetnum
    return exdata

# 因为此数据集中的year，month, day, hour均为整数型，再程序中会被当做连续的变量

# 删除原始数据中不需要的字段名
def Shanchu(exdata, aiduan=['No']):
    for ai in aiduan:
        if ai in exdata.keys():
            del exdata[ai]
    return exdata


# 因此这里将这4个变量变为字符串的形式
def TeshuHandle(exdata, ziduan=['year', 'month', 'day', 'hour'], tianjiastr=['年', '月', '日', '时']):
    for j, k in zip(ziduan, tianjiastr):
        if j in exdata.keys():
            exdata[j] = ['%d%s' % (j, k) for j in exdata[j]]
    return exdata

# 数据处理后最终的数据集

first = DeleteTargetNan(data, 'pm2.5')
two = Shanchu(first)
third = TeshuHandle(two)

# 将数据集按照7；1.5:1.5的比例分为训练，测试、预测数据集

def fenge(exdata, per=[0.15, 0.15]):
    # 总长度
    lent = len(exdata)
    alist = np.arange(lent)
    np.random.shuffle(alist)

    # 验证
    shu = int(lent * per[0])
    yu = int(lent * per[1])

    yanzheng = np.random.choice(alist, shu)

    # 预测
    shengxai = np.array([i for i in alist if i not in yanzheng])

    yuce = np.random.choice(shengxai, yu)

    # 训练
    train = np.array([j for j in alist if j not in yanzheng and j not in yuce])

    # 返回数据集
    dadata = {}
    dadata[0] = {}

    dadata[0]['train'] = exdata[train]
    dadata[0]['test'] = exdata[yanzheng]

    return dadata, exdata[yuce]

deeer = fenge(third.values)

# 数据
dt_data = deeer[0]
test_data = deeer[1]


#  数据结构和决策树分类是相同的，
