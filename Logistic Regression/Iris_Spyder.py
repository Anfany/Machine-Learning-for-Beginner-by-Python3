#-*- coding：utf-8 -*-
# &Author  AnFany

from bs4 import BeautifulSoup as bs
import urllib


html = urllib.request.urlopen("https://en.wikipedia.org/wiki/Iris_flower_data_set#Data_set")

bsObj = bs(html.read(), "html5lib")

nameList = bsObj.findAll('table',{'class' : "wikitable sortable mw-collapsible mw-collapsed"})

datalist = str(nameList[0].get_text()).split('\n')


data_set_name = datalist[1]

#字段名称
att_name = [fu.replace(' ','_') for fu in datalist[3:9]]

for hh in att_name:
    exec('%s = []'%hh)

#开始获得数据
for dat in range(len(datalist[9:-2])):
    if dat % 8 > 1:
        if dat % 8 != 7:
            eval(att_name[dat % 8 - 2]).append(datalist[9:-2][dat])
        else:

            eval(att_name[dat % 8 - 2]).append(datalist[9:-2][dat].split('.')[1][1:])



#有序的字典形式,按添加的序列输出
from collections import OrderedDict
datadict=OrderedDict({})

for keyname in att_name:
    datadict[keyname] = eval(keyname)



#写入文件
import pandas as pd
df = pd.DataFrame(datadict)
df.to_csv(r'C:\Users\GWT9\Desktop\Iris.csv', index=False)
print('完毕')

