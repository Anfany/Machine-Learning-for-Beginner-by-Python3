#-*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd

data = pd.read_csv(r'C:\Users\GWT9\Desktop\iris.csv')

# y值Softmax
ydata = data['Species'].values
# x值
xdata = data.iloc[:, 1:5].values

# 数据处理
import numpy as np

# x数据标准化
handle_x_data = (xdata - np.mean(xdata, axis=0)) / np.std(xdata, axis=0)

# y数据独热化
ydata = pd.get_dummies(data['Species']).values

# 因为数据中类别比较集中，不易于训练，因此打乱数据

# 首先将x数据和y数据合在一起
xydata = np.hstack((handle_x_data, ydata))
# 打乱顺序
np.random.shuffle(xydata)

# 分离数据

X_DATA = xydata[:, :4]

Y_DATA = xydata[:, 4:]

Data = [X_DATA, Y_DATA]

# 数据结构

# X_DATA.shape = (样本数, 特征数)
# Y_DATA.shape = (样本数, 类别数)

# 类别
# setosa  [1,0,0]
# versicolor [0,1,0]
# virginica  [0,0,1]



