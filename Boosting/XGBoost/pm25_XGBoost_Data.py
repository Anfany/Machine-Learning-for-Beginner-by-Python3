# -*- coding：utf-8 -*-
# &Author  AnFany



import pandas as pd
import numpy as np

# 读取数据文件
data = pd.read_csv(r'C:\Users\GWT9\Desktop\PRSA_data_2010.1.1-2014.12.31.csv')


'''第一部分：缺失值的处理'''
#  因为Pm2.5是目标数据，如有缺失值直接删除这一条记录

# 删除目标值为空值的行的函数, 其他列为缺失值则自动填充的函数,并将目标变量放置在数据集最后一列
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



# 删除原始数据中不需要的字段名
def Shanchu(exdata, aiduan=['No']):
    for ai in aiduan:
        if ai in exdata.keys():
            del exdata[ai]
    return exdata


# 将数据中的属性值为字符串的进行数字编码，因为独热编码对决策树而言不那么重要
def Digit(eadata):
    # 判断是字符串
    for jj in eadata:
        try:
            eadata[jj].values[0] + 1
        except TypeError:
            # 需要转为数字编码
            numlist = list(set(list(eadata[jj].values)))
            zhuan = [numlist.index(jj) for jj in eadata[jj].values]
            eadata[jj] = zhuan
    return eadata


# 数据处理后最终的数据集

first = DeleteTargetNan(data, 'pm2.5')
two = Shanchu(first)
third = Digit(two)

# 将数据集按照8:2的比例分为训练、预测数据集。其中训练数据集再分为K份，进行K折交叉验证
# 为了便于确定最有参数，在这里把数据集固定下来，也就是，哪些作为最终的预测数据是固定的

# 每一折交叉数据集中，训练和测试的数据集也是确定的

def fenge(exdata, k=10, per=[0.8, 0.2]):
    # 总长度
    lent = len(exdata)
    alist = np.arange(lent)
    np.random.shuffle(alist)

    # 训练
    xunlian_sign = int(lent * per[0])

    xunlian = np.random.choice(alist, xunlian_sign, replace=False)

    # 预测
    yuce = np.array([i for i in alist if i not in xunlian])

    # 再将训练数据集分为K折
    # 存储字典
    save_dict = {}
    for jj in range(k):
        save_dict[jj] = {}
        length = len(xunlian)
        # 随机选
        yuzhi = int(length / k)
        yan = np.random.choice(xunlian, yuzhi, replace=False)
        tt = np.array([i for i in xunlian if i not in yan])
        save_dict[jj]['train'] = exdata[tt]
        save_dict[jj]['test'] = exdata[yan]

    return save_dict, exdata[yuce]

deeer = fenge(third.values)

# K折交叉的训练数据
dt_data = deeer[0]
# 预测数据
predict_data = deeer[1]