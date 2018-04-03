#-*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd

data = pd.read_csv(r'C:\Users\GWT9\Desktop\Boston.csv')

#y值
y_data = data['MEDV']
#x值
x_data = data.drop('MEDV', axis=1).values

#如果x值只有一个特征
#x_data= data.drop('MEDV', axis=1).values.reshape(-1,1)

#对x值进行[0,1]归一化
from sklearn import preprocessing as spp #引入数据预处理的库
scaler_01 = spp.MinMaxScaler()

#归一的x值
x_pre_data = scaler_01.fit_transform(x_data)

#归一的x值，y值分为训练数据集和预测数据集
import numpy as np
def divided(xdata, ydata, percent=0.1):
    sign_list = list(range(len(xdata)))
    #用于测试的序号
    select_sign = sorted(np.random.choice(sign_list, int(len(x_data)*percent), replace=False))

    #用于训练的序号
    no_select_sign = [isign for isign in sign_list if isign not in select_sign]

    #测试数据
    x_predict_data = xdata[select_sign]
    y_predict_data = ydata[select_sign].values.reshape(len(select_sign),1)#转化数据结构

    #训练数据
    x_train_data = xdata[no_select_sign]
    y_train_data = ydata[no_select_sign].values.reshape(len(no_select_sign),1)#转化数据结构

    return x_train_data, y_train_data, x_predict_data, y_predict_data #训练的x，y;  测试的x，y

#可用于算法的数据
model_data = divided(x_pre_data, y_data)


#数据结构
#x_train_data.shape = (样本数，特征数)
#y_train_data.shape= (样本数，1)



