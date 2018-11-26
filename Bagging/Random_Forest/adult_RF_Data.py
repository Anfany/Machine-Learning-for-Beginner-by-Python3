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

#  将字符串字段进行数字编码，
def Digitcode(traindata, predixdata):
    #  数字编码
    for ikey in traindata:
        if traindata[ikey].dtype == 'object':  # 数字编码
            numb = sorted(list(set(list(traindata[ikey].values))))
            exdict = {ji: numb.index(ji) for ji in numb}
            if ikey == 'Money':  # 因为sklearn支持字符串类别
                predixdata[ikey] = ['%s' % gfi[:-1] for gfi in predixdata[ikey]]  # 因为测试数据文件中Money的值多个点
            else:
                predixdata[ikey] = [exdict[fi] for fi in predixdata[ikey]]
                traindata[ikey] = [exdict[fi] for fi in traindata[ikey]]
    return traindata, predixdata.values


# 读取的数据
read_train = ReadHandle(train_path)
read_pre = ReadHandle(pre_path)

# 经过处理的数据
han_train, predict_data = Digitcode(read_train, read_pre)


#  将训练数据进行K折交叉验证，根据F1度量确定最佳的
#  然后再进行预测数据的计算，输出混淆矩阵以及精确率、召回率，F1度量

def kfold(trdata, k=10):
    vadata = trdata.values
    legth = len(vadata)
    datadict = {}
    signnuber = np.arange(legth)
    for hh in range(k):
        datadict[hh] = {}
        np.random.shuffle(vadata)
        yanzhneg = np.random.choice(signnuber, int(legth / k), replace=False)
        oneflod_yan = vadata[yanzhneg]
        oneflod_xun = vadata[[hdd for hdd in signnuber if hdd not in yanzhneg]]
        # 训练数据和验证数据
        datadict[hh]['train'] = oneflod_xun
        datadict[hh]['test'] = oneflod_yan
    return datadict

#  存储K折交叉验证的数据字典
dt_data = kfold(han_train)
