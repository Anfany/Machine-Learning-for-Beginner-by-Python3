#-*- coding：utf-8 -*-
# &Author  AnFany

from bs4 import BeautifulSoup as bs
import urllib

html = urllib.request.urlopen("http://lib.stat.cmu.edu/datasets/boston")
bsObj = bs(html.read(), "html5lib")
namest = str(bsObj.body.get_text())


sign = 0
name = []
for hh in namest.split('\n'):
    if 6< sign < 21:
        #获取字段名字
        name.append(hh[:7].replace(' ',''))

    if sign == 21:
        for nam in name:
            exec('%s = []'%nam)

    if sign > 21:
        #获得字段对应的数据
        datalist = [data for data in hh.split(' ') if data !='']
        if sign % 2 ==0:
            for i in range(len(datalist)):
                eval(name[i]).append(datalist[i])
        else:
            for i in range(len(datalist)):
                eval(name[11+i]).append(datalist[i])
    sign += 1

#有序的字典形式,按添加的序列输出
from collections import OrderedDict
datadict=OrderedDict({})

for keyname in name:
    datadict[keyname] = eval(keyname)


#写入文件
import pandas as pd
df = pd.DataFrame(datadict)
df.to_csv(r'C:\Users\GWT9\Desktop\Boston.csv', index=False)
print('完毕')
