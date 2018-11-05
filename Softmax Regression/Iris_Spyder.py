#-*- coding：utf-8 -*-
# &Author  AnFany

from bs4 import BeautifulSoup as bs
import urllib


html = urllib.request.urlopen("https://en.wikipedia.org/wiki/Iris_flower_data_set#Data_set")

bsObj = bs(html.read(), "html5lib")


# 字段列表
ziduan = []
for hh in bsObj.find_all('table', class_='wikitable sortable mw-collapsible mw-collapsed'):
    for ii in hh.find_all('th'):
        fu = ii.get_text().split()
        zi = ('_').join(fu)
        exec('%s = []' % zi)
        ziduan.append(zi)
    fu = 0
    for jj in hh.find_all('td'):
        ty = jj.get_text().split()
        try:
            float(ty[0])
            exec('%s.append(%.2f)' % (ziduan[fu % 6], float(ty[0])))
        except ValueError:
            exec('%s.append("%s")' % (ziduan[fu % 6], str(ty[-1])))
        fu += 1


#有序的字典形式,按添加的序列输出
from collections import OrderedDict
datadict=OrderedDict({})

for keyname in ziduan:
    datadict[keyname] = eval(keyname)



#写入文件
import pandas as pd
df = pd.DataFrame(datadict)
df.to_csv(r'C:\Users\GWT9\Desktop\Iris.csv', index=False)
print('完毕')

