# -*- coding：utf-8 -*-
# &Author  AnFany


# 实现切割图片，并返回空间金字塔池化的结果

from skimage import io
from PIL import Image
import numpy as np


class CutFig():

    def __init__(self):
        # 需要切割的图片的路径
        self.fig_path = r"C:\Users\GWT9\Desktop\iris_1.jpg"   # 要处理的图片路径

        # 切割线的宽度为多少像素
        self.cut_width = 3

        # 可以自定义切割线的颜色的RGB字典
        self.cut_color_dict = {'R': 255, 'G': 255, 'B': 255}

        # 该图片3个通道的数字矩阵
        self.R, self.G, self.B = self.get_matrix()

        # 空间金字塔池化的层数列表，1层得到的值的个数为1， 2层得到的值的个数为4，n层得到的值得个数为n^2
        self.layer_list = [1, 2, 3, 4]

        # 保存切割图片的路径
        self.cut_fig_path = r"C:\Users\GWT9\Desktop"  # 保存的切割图片的路径

    # 获得图片的三个通道的数字矩阵
    def get_matrix(self):
        matrix = io.imread(self.fig_path)
        R = matrix[:, :, 0]  # 获取R通道
        G = matrix[:, :, 1]  # 获取G通道
        B = matrix[:, :, 2]  # 获取B通道
        return R, G, B

    # 三个通道合成的矩阵转变为图片
    def generate_fig(self, r, g, b, title, l):
        """
        :param r: R通道矩阵
        :param g: G通道矩阵
        :param b: B通道矩阵
        :param title: 图片基础名称
        :param l: 层标识
        :return: 以图片基础名称_层标识为名的png格式的图片
        """
        # 通道矩阵变换维度
        R = r.reshape(-1, len(r[0]), 1)
        G = g.reshape(-1, len(g[0]), 1)
        B = b.reshape(-1, len(b[0]), 1)
        # 三个矩阵合成为三维矩阵
        fig_matrix = np.concatenate([R, G, B], 2)
        image = Image.fromarray(np.uint8(fig_matrix))
        image.show()
        image.save(r"%s\%s_%s.png" % (self.cut_fig_path, title, l))
        return print('图片生成完毕')

    def ssp_pooling(self, channel_name, l):
        """
        根据层数来对单一通道的数字矩阵进行分割，并且返回空间金字塔池化的数值
        :param channel_name: 单一通道的数字矩阵
        :param l: 层数
        :return: 分割后的单通道的数字矩阵以及池化得到的值
        """
        count = l
        # 分割都是均分，对于不能均分的情况，例如行列不能被count整除时，最后的稍微大一些
        # 如果count 大于行或者列，则不进行任何的操作
        channel_matrix = eval('self.%s' % channel_name)
        if count == 1:
            print('第%s层的%s通道ssp结果:' % (l, channel_name), np.max(channel_matrix))
            return channel_matrix

        else:
            row, column = channel_matrix.shape

            if count > row or count > column:
                return print('层数过大')
            else:
                # 每一块行、列的像素数
                row_length = row // count
                column_length = column // count
                # 分割后的图片就是先在单通道数字矩阵的周围添加分割线
                # 添加分割线后的数字矩阵的尺寸大小，初始的值就设为分割线定义的该通道的数字
                cut_fig_channel_matrix = np.ones((row + (count - 1) * self.cut_width,
                                                  column + (count - 1) * self.cut_width)) * \
                                         self.cut_color_dict[channel_name]

                # 存储ssp的结果
                ssp_result = []
                for r in range(count):
                    for c in range(count):
                        # 行的范围
                        if row < (r + 2) * row_length:
                            row_field = [r * row_length, (r + 2) * row_length]
                        else:
                            row_field = [r * row_length, (r + 1) * row_length]
                        # 列的范围
                        if column < (c + 2) * column_length:
                            column_field = [c * column_length, (c + 2) * column_length]
                        else:
                            column_field = [c * column_length, (c + 1) * column_length]
                        # 获取数字矩阵块
                        group_matrix = channel_matrix[row_field[0]: row_field[1], column_field[0]: column_field[1]]

                        # 获取池化结果
                        ssp_result.append(np.max(group_matrix))

                        # 因为添加分割线的矩阵中的初始值已经定义好，把块里面的数字覆盖为原始图片的数字即可
                        # 行的范围
                        if row < (r + 2) * row_length:
                            cut_row_field = [r * row_length + r * self.cut_width,
                                             (r + 2) * row_length + r * self.cut_width]
                        else:
                            cut_row_field = [r * row_length + r * self.cut_width,
                                             (r + 1) * row_length + r * self.cut_width]
                        # 列的范围
                        if column < (c + 2) * column_length:
                            cut_column_field = [c * column_length + c * self.cut_width,
                                                (c + 2) * column_length + c * self.cut_width]
                        else:
                            cut_column_field = [c * column_length + c * self.cut_width,
                                                (c + 1) * column_length + c * self.cut_width]

                        cut_fig_channel_matrix[cut_row_field[0]: cut_row_field[1], cut_column_field[0]: cut_column_field[1]] = group_matrix
                print('第%s层的%s通道ssp结果:' % (l, channel_name), ssp_result)
                return cut_fig_channel_matrix




# 最终的主函数
if __name__ == "__main__":

    c_f_ssp = CutFig()
    for l in c_f_ssp.layer_list:
        # 获取R通道
        cut_r = c_f_ssp.ssp_pooling('R', l)
        # 获取G通道
        cut_g = c_f_ssp.ssp_pooling('G', l)
        # 获取B通道
        cut_b = c_f_ssp.ssp_pooling('B', l)

        # 生成图片
        c_f_ssp.generate_fig(cut_r, cut_g, cut_b, 'horizontal_iris', l)
