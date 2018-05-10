# -*- coding：utf-8 -*-
# &Author  AnFany

import urllib.request
from collections import OrderedDict
import pandas as pd
#属性名称
att_name = ['Age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',\
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',\
            'hours-per-week', 'native-country']

#类别名称
type_name ='Money'

# 训练数据网址
train_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# 测试数据网址
test_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


# 构建函数
def spyderfunc(url, dataname, aname=att_name, tname=type_name):

    # 有序的字典形式,按添加的序列输出
    datadict = OrderedDict({})
    for keyname in aname:
        datadict[keyname] = []
    datadict[tname] = []

    # 打开网页，读取内容
    html = urllib.request.urlopen(url)

    namest = html.read().decode('utf-8')
    for hh in namest.split('\n'):
        if len(hh) > 30:
            hh = hh.replace(' ', '')# 去掉空格
            hang = hh.split(',')
            for jjj in range(len(hang)):
                try:
                    datadict[att_name[jjj]].append(hang[jjj])
                except IndexError:
                    datadict[type_name].append(hang[jjj])
        elif 6 < len(hh) < 30:
            pass
        else:
            break
    #写入文件
    df = pd.DataFrame(datadict)
    df.to_csv(r'C:\Users\GWT9\Desktop\Adult_%s.csv'%dataname, index=False)
    print('完毕')


# 训练数据
spyderfunc(train_url, 'Train')

# 测试数据
spyderfunc(test_url, 'Test')
