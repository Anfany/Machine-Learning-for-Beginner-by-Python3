# -*- coding：utf-8 -*-
# &Author  AnFany

#  引入库

import imageio  # 引入合成gif的库
import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体为新宋体

'''
输出的规则：
一、格式方面
1.str，同时打印
2.[str]，单层列表，一个接一个的打印，字体为红色
3.[[str]]，双层列表，按序依次打印，字体为红色

二、位置方面
1. 前一项是str，则width不变，height减小
2. 前一项是list，则width增大，height不变
'''

wenzi = ['对你来说', [['关注']], '，是0.6秒的事', [['点赞']], '，是0.2秒的事', [['分享']], '，是0.3秒的事', '但对我而言',\
         '这都是', ['一辈子'], '的事!!!']

# 先把语句拼接起来
def yuju(wei, width, height):
    swi = width
    fu = []
    uu = 0
    for jj in wei:
        if type(jj) == str:
            if uu == 1:
                height += 0.02
            elif uu == 0:
                width = swi
            fu.append(['plt.text(%.2f, %.2f, "%s")'%(width, height, jj)])
            height -= 0.02
            uu = 0
        elif type(jj) == list:
            width = swi
            for hh in jj[0]:
                fu.append(['plt.text(%.2f, %.2f, "%s", color = "r")' % (width, height, hh)])
                width += 0.06
                count = len(hh)
            width += (count - 1) * 0.06
            height -= 0.02
            uu = 1
    return fu

#  绘制每个子图
def fig(exlist):
    plt.figure(figsize=(2.2, 2.0))
    plt.xlim(0.2, 0.8)
    plt.ylim(0.79, 0.9)
    gu = 0
    for jj in exlist:
        plt.axis('off')
        for hh in jj:
            eval(hh)
        plt.savefig(r"C:\Users\GWT9\Desktop\%s.png"%gu)
        gu += 1
    return gu

fufu = yuju(wenzi, 0.2, 0.9)

figu = fig(fufu)

namelist = ['%s.png'%dd for dd in range(figu)]

#  合成一个gif图片
def create_gif(image_list, gif_name='anfany.gif'):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.9)  # duration控制动态图中每张图片的显示时间

import os
os.chdir(r'C:\Users\GWT9\Desktop')

create_gif(namelist)
