'''
输入图像数据(N,C,H,W)，N是迷你批次的大小，C是图像的通道数，H是图像的高，W是图像的宽

1、使用了im2col函数(image to column)将多维数据展开为二维数据，然后使用矩阵运算就能完成卷积运算。

'''
import numpy as np

def im2col(input_date,filter_h,filter_w,stride=1,padding=0):
    """
    :parameter:
    input_data : 思维数据构成的输入数据（迷你批次数，通道，高，宽）
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    padding : 填充

    :return:
    col ： 输出二维数组
    """
    N, C, H, W = input_date.shape
    # 计算输出数据的高和宽(out_h,out_w)
    out_h = (H+2*padding-filter_h)//stride+1
    out_w = (W+2*padding-filter_w)//stride+1
    # 对输入数据的边缘进行填充处理
    img = np.pad(input_date)
    # 在图的数据上滑动取值



if __name__=="__main__":
    pass