# -*- coding：utf-8 -*-
# &Author  AnFany


import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\GWT9\Desktop\PRSA_data_2010.1.1-2014.12.31.csv')

# 数据处理
# 目标字段是pm2.5，因此删去此列为NaN的行
data_nan = data[np.isfinite(data['pm2.5'])]

# 第一个字段是序号字段，不需要
data_one = data_nan[data_nan.columns[1:]]

# 字段'cbwd'，独热编码

one_data = pd.get_dummies(data_one['cbwd'], prefix='cbwd')

# 删除原来的字段'cbwd'

data_cw = data_one.drop(['cbwd'], axis=1)

# 添加上独热产生的数据
data_hh = pd.concat([data_cw, one_data], axis=1)

#  获得目标数据的最大与最小值，


ymax = np.max(data_hh['pm2.5'].values, keepdims=True)

ymin = np.min(data_hh['pm2.5'].values, keepdims=True)


# 所有特征数据标准化， 目标数据0-1化
def norm(dat):
    da = pd.DataFrame()
    for hh in dat.columns:
        if hh != 'pm2.5':
            da[hh] = (dat[hh] - np.mean(dat[hh])) / np.std(dat[hh])  # 标准化
            #  da[hh] = (dat[hh] - np.min(dat[hh])) / (np.max(dat[hh]) - np.min(dat[hh])) # 0-1化
        else:
            da[hh] = (dat[hh] - np.min(dat[hh])) / (np.max(dat[hh]) - np.min(dat[hh]))  # 0-1化
    return da

datee = norm(data_hh)


# 目标数据和特征数据分离

Ydata = np.array(datee['pm2.5'].values).reshape(-1, 1)  # 目标数据
Xdata = datee.drop(['pm2.5'], axis=1).values  # 特征数据


#  训练数据分为测试数据和预测数据
def divided(xdata, ydata, percent=0.3):
    sign_list = list(range(len(xdata)))
    #  用于测试的序号
    select_sign = np.random.choice(sign_list, int(len(xdata) * percent), replace=False)

    #  用于训练的序号
    no_select_sign = [isign for isign in sign_list if isign not in select_sign]

    # 测试数据
    x_predict_data = xdata[select_sign]
    y_predict_data = np.array(ydata[select_sign]).reshape(-1, len(ydata[0]))  # 转化数据结构

    # 训练数据
    x_train_data = xdata[no_select_sign]
    y_train_data = np.array(ydata[no_select_sign]).reshape(-1, len(ydata[0]))  # 转化数据结构

    return x_train_data, y_train_data, x_predict_data, y_predict_data # 训练的x，y;  测试的x，y;


# 可用于算法的数据
model_data = list(divided(Xdata, Ydata))
model_data.append([ymax, ymin])

# 数据结构

# 输入数据 (样本数，特征数）
# 输出数据 (样本数，输出维度）

