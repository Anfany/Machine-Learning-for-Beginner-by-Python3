#-*- coding：utf-8 -*-
# &Author  AnFany

from bs4 import BeautifulSoup as bs
import urllib

html = urllib.request.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
bsObj = bs(html.read(), "html5lib")
namest = str(bsObj.body.get_text())


# 属性名称
att_name = ['Wine Type', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', \
            'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',\
            'Color intensity', 'Hue', 'OD280/OD315', 'Proline']

# 构建字典
# 有序的字典形式,按添加的序列输出
from collections import OrderedDict
datadict=OrderedDict({})
for keyname in att_name:
    datadict[keyname] = []

# 添加数据
for hh in namest.split('\n'):
    sample = hh.split(',')
    if len(sample) > 1:
        for hhh in range(len(sample)):
            if hhh == 0:
                datadict[att_name[hhh]].append(int(sample[hhh]))
            else:
                datadict[att_name[hhh]].append(float(sample[hhh]))
    else:
        break

# 写入文件
import pandas as pd
df = pd.DataFrame(datadict)
df.to_csv(r'C:\Users\GWT9\Desktop\Wine.csv', index=False)
print('完毕')



