'''
输入图像数据(N,C,H,W)，N是迷你批次的大小，C是图像的通道数，H是图像的高，W是图像的宽

1、使用了im2col函数(image to column)将多维数据展开为二维数据，然后使用矩阵运算就能完成卷积运算。
2、使用col2im函数将反向传播回来的数据从二维变为与图像维度一直大数据(N,C,H,W)
'''
import numpy as np

def im2col(input_date, filter_h, filter_w, stride=1, padding=0):
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
    # 对输入数据的边缘进行填充处理(填充0)
    img = np.pad(input_date,((0,0),(0,0),(padding, padding),(padding, padding)),'constant')
    col = np.zeros(N,C,filter_h,filter_w,out_h,out_w)
    # 在图的数据上滑动取值
    for x in range(filter_h):
        x_max = x+stride*out_h
        for y in range(filter_w):
            y_max = y+stride*out_w
            col[:,:,x,y,:,:] = img[:,:,x:x_max:stride,y:y_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col

def col2im():
    pass


if __name__=="__main__":

    pass

