# -*- coding：utf-8 -*-
# &Author  AnFany


# 实现最大值池化、2种形式的均值池化
# 对于超出范围的只计算范围内的

import numpy as np

Fig_matrix = np.array([[-2, -6, -1, -2, 0, 8, 3],
                       [-5, -16, -5, -7, -3, 23, 13],
                       [-7, -18, -12, -8, -7, 26, 26],
                       [-14, -13, -11, -6, -8, 19, 33],
                       [-26, -9, 7, -5, -7, 14, 26],
                       [-34, -12, 25, 0, -10, 12, 19],
                       [-27, -13, 22, 3, -14, 10, 19],
                       [-9, -5, 7, 1, -7, 4, 9]])


class Pool():

    def __init__(self):
        # 池化窗口的尺寸
        self.p_size = 2

        # 池化窗口的步长
        self.p_strides = 3

        # 均值方法，普通均值为1，其他为加权均值
        self.method = 1

    # 最大值池化
    def max_pooling(self, fig):
        """
        实现最大值池化
        :param fig: 数字矩阵
        :return: 池化后的矩阵
        """
        row, column = fig.shape
        pool_row = (row - 1) // self.p_strides + 1
        pool_column = (column - 1) // self.p_strides + 1
        pool_matrix = np.zeros((pool_row, pool_column))
        for i in range(pool_row):
            for j in range(pool_column):
                pool_matrix[i, j] = np.max(
                    fig[i * self.p_strides: i * self.p_strides + self.p_size,
                    j * self.p_strides: j * self.p_strides + self.p_size])
        return pool_matrix

    # 均值池化
    def mean_pooling(self, fig):
        """
        实现最大值池化
        :param fig: 数字矩阵
        :return: 池化后的矩阵
        """
        row, column = fig.shape
        pool_row = (row - 1) // self.p_strides + 1
        pool_column = (column - 1) // self.p_strides + 1
        pool_matrix = np.zeros((pool_row, pool_column))
        # 普通均值
        if self.method == 1:
            for i in range(pool_row):
                for j in range(pool_column):
                    pool_matrix[i, j] = np.mean(
                        fig[i * self.p_strides: i * self.p_strides + self.p_size,
                        j * self.p_strides: j * self.p_strides + self.p_size])
        else:
            # 加权均值
            for i in range(pool_row):
                for j in range(pool_column):
                    # 在范围内的数字
                    in_matrix = fig[i * self.p_strides: i * self.p_strides + self.p_size,
                               j * self.p_strides: j * self.p_strides + self.p_size]
                    # 归一化到01之间
                    norm_matrix = (in_matrix - np.min(in_matrix)) / (np.max(in_matrix) - np.min(in_matrix))
                    # 计算各个的权重
                    norm_matrix = norm_matrix / np.sum(norm_matrix)
                    # 然后计算加权均值
                    pool_matrix[i, j] = np.sum(in_matrix * norm_matrix)
        return pool_matrix


# 最终的主函数
if __name__ == "__main__":
    # 原始图片路径
    p = Pool()
    print(p.max_pooling(Fig_matrix))





