# -*- coding：utf-8 -*-
# &Author  AnFany


import pandas as pd
import numpy as np

#  训练数据文件路径
train_path = 'C:/Users/GWT9\Desktop/Adult_Train.csv'

#  预测数据文件路径
pre_path = 'C:/Users/GWT9\Desktop/Adult_Test.csv'

# 读取并且处理确缺失数据
def ReadHandle(filepath, miss='fill'):  # 定义处理数据的函数
    data = pd.read_csv(r'%s' % filepath)
    data = data.replace('?', np.nan)
    #  处理缺失值
    if miss == 'del':  # 删除掉缺失值
        miss_data = data.dropna(how='any')
    else:
        miss_data = data.fillna(method='ffill')
    return miss_data

#  因为CatBoost支持字符串和数字的特征值

# 读取的数据
read_trai = ReadHandle(train_path)
read_pr = ReadHandle(pre_path)

# 因为目标字段"Money"中，预测数据较训练数据多了一个点，需要处理
read_pr["Money"] = [ii[:-1] for ii in read_pr["Money"]]


# 需要将目标字段Money中的值转换为0和1
exdixxt = {'<=50K': 0, '>50K': 1}

# 定义将目标字段值数字化的函数
def Tran(data, di=exdixxt):
    data['Money'] = [di[hh] for hh in data['Money']]
    return data


# 目标字段值转换
read_train = Tran(read_trai)
read_pre = Tran(read_pr)


# 将训练数据按照比例分为训练和验证数据，
def fenge(datd, bili=0.2): # 80%的数据用于训练，20%用于验证
    # 随机打乱数据
    np.random.seed(1900)
    np.random.shuffle(datd)
    # 获取总长度
    leng = len(datd)
    # 验证需要的
    yan_sign = int(leng * bili)

    # 为了便于验证参数组合，每一次都是固定的
    np.random.seed(100)
    yan_list = np.random.choice(list(range(leng)), yan_sign, replace=False)

    # 用于训练的
    xunlian_list = [gg for gg in list(range(leng)) if gg not in yan_list]

    # 存储数据的字典
    sadic = {}
    sadic['train'] = datd[xunlian_list]

    sadic['test'] = datd[yan_list]
    return sadic

# 获得类别型特征的索引
def Catindex(data):
    catlist = []
    sign = 0
    for jj in read_train:
        if jj != 'Money':
            try:
                read_train[jj][0] + 1
            except TypeError:
                catlist.append(sign)
        sign += 1
    return catlist


datt = fenge(read_train.values)
datt['predict'] = read_pre.values
catind = Catindex(read_train)

