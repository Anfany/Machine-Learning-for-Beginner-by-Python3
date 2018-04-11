#-*- coding：utf-8 -*-
# &Author  AnFany

import pandas as pd

data = pd.read_csv(r'C:\Users\GWT9\Desktop\Wine.csv')

# y值Softmax
y_data = data['Wine Type'].values
# x值
xdata = data.iloc[:, 1:].values


# 数据处理
import numpy as np

# x数据标准化
x_data = (xdata - np.mean(xdata, axis=0)) / np.std(xdata, axis=0)


DATA = [x_data, y_data]

# 数据结构
# x_data.shape = (样本数, 特征数)
# y_data.shape = (样本数,)


