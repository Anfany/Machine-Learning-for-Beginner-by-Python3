#-*- coding：utf-8 -*-
# &Author  AnFany

from bs4 import BeautifulSoup as bs
import urllib

html = urllib.request.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
bsObj = bs(html.read(), "html5lib")
namest = str(bsObj.body.get_text())

#属性名称
att_name = ['Age','Sex','Chest_Pain_Type','Rest_Blood_pre','Serum_Cho','Fast_Blood_Sugar',\
            'Rest_Ele_Result','Max_Heart_Rate','Exercise_Angina','OldPeak','ST','Major_Vess',\
            'Thal']
#类别名称
type_name ='Heart_Disease'

#构建字典
#有序的字典形式,按添加的序列输出
from collections import OrderedDict
datadict=OrderedDict({})
for keyname in att_name:
    datadict[keyname] = []
datadict[type_name] =[]

#添加数据
for hh in namest.split('\n'):
    sample = hh.split(' ')
    if len(sample) > 1:
        for hhh in range(len(sample)):
            try:
                datadict[att_name[hhh]].append(float(sample[hhh]))
            except IndexError:
                datadict[type_name].append(sample[hhh])
    else:
        break

#写入文件
import pandas as pd
df = pd.DataFrame(datadict)
df.to_csv(r'C:\Users\GWT9\Desktop\Heart.csv', index=False)
print('完毕')
