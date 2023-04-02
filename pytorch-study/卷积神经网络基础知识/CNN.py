# wuyong@ccnu
# 学习卷积神经网络的运行原理
# 学习原理----学习使用PyTorch
# 从小处入手，然后让程序慢慢长大
'''
框架代码：
1、初始化函数----设定卷积层，池化层，线性层
2、训练----学习给定训练集样本后、优化权重
3、查询----给定输入，从输出节点给出答案

CNN中传递的数据是4维数据。其形状如(10,1,28,28),对应10个高为28，长为28
通道为1的数据

使用了im2col函数(image to column)将多维数据展开为二维数据，然后使用矩阵运算就能完成卷积运算。
'''

import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self):

        pass

    def forward(self):

        pass 

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """

        Parameters
        ----------
        input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
        filter_h : 滤波器的高
        filter_w : 滤波器的长
        stride : 步幅
        pad : 填充

        Returns
        -------
        col : 2维数组
        """

        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col


    pass