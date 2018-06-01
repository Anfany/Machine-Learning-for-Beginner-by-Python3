# -*- coding：utf-8 -*-
# &Author  AnFany


import pandas as pd
import numpy as np

#  训练数据文件路径
train_path = 'C:/Users/GWT9\Desktop/Adult_Train.csv'

#  测试数据文件路径
test_path = 'C:/Users/GWT9\Desktop/Adult_Test.csv'

#  因为测试数据native-country中不存在Holand-Netherlands，不便于独热编码。
#  因此在测试文件中添加一个native-country为Holand-Netherlands的样本，然后在删除即可
#  为简化程序，手动添加
def handle_data(filepath, miss='fill'):  # 定义处理数据的函数
    data = pd.read_csv(r'%s'%filepath)
    data = data.replace('?', np.nan)
    #  处理缺失值
    if miss == 'del':  # 删除掉缺失值
        miss_data = data.dropna(how='any')
    else:
        miss_data = data.fillna(method='ffill')
    #  新建DataFrame
    newdata = pd.DataFrame()
    #  独热化编码
    for ikey in miss_data:
        if miss_data[ikey].dtype == 'object':  # 独热编码
            onedata = pd.get_dummies(miss_data[ikey])
            newdata = pd.concat([newdata, onedata], axis=1)
        else:
            newdata[ikey] = miss_data[ikey]
    return newdata


train_data = handle_data(train_path)
test_data = handle_data(test_path)
test_data = test_data.drop([len(test_data) - 1], inplace=False)  # 删除添加的最后一个样本


#  数据标准化
# 所有特征数据标准化， 目标数据0-1化
def norm(trdata, tedata):
    tr_da = pd.DataFrame()
    te_da = pd.DataFrame()
    for hh in trdata.columns:
        if hh not in ['<=50K', '>50K']:
            tr_da[hh] = (trdata[hh] - np.mean(trdata[hh])) / np.std(trdata[hh])  # 标准化
            te_da[hh] = (tedata[hh] - np.mean(trdata[hh])) / np.std(trdata[hh])  # 标准化
            #  tr_da[hh] = (trdata[hh] - np.min(trdata[hh])) / (np.max(trdata[hh]) - np.min(trdata[hh])) # 0-1化
            #  te_da[hh] = (tedata[hh] - np.min(trdata[hh])) / (np.max(trdata[hh]) - np.min(trdata[hh]))  # 0-1化
        else:
            tr_da[hh] = trdata[hh].values
            te_da[hh] = tedata['%s.'%hh].values  # 训练数据和测试数据的Money字段内容不同。测试数据的多个"."
    return tr_da, te_da


Train_data, Test_data = norm(train_data, test_data)

#  将训练数据平均分为n份，利用K折交叉验证计算模型最终的正确率
#  将训练数据分为训练数据和验证数据

def kfold(trdata, k=10):
    vadata = trdata.values
    legth = len(vadata)
    datadict = {}
    signnuber = np.arange(legth)
    for hh in range(k):
        np.random.shuffle(vadata)
        yanzhneg = np.random.choice(signnuber, int(legth / k), replace=False)
        oneflod_yan = vadata[yanzhneg]
        oneflod_xun = vadata[[hdd for hdd in signnuber if hdd not in yanzhneg]]
        datadict[hh] = [oneflod_xun, oneflod_yan]
    return datadict

#  存储K折交叉验证的数据字典
kfold_train_datadict = kfold(Train_data)


