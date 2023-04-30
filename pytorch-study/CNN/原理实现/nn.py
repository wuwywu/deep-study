'''
输入图像数据(N,C,H,W)，N是批次中的数量，C是图像的通道数，H是图像的高，W是图像的宽

1、卷积层：Conv2d卷积方向是二维的。
2、池化层：Pooling没有学习参数，运算后通道不发生改变

'''
import numpy as np
from util import im2col

class Conv2d:
    def __init__(self, W, b, stride=1, padding=0):
        # w : 卷积核的大小（FN, FC, FH, FW）
        # b : 大小为FN
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b   # 卷积运算(N*out_h*out_w,FN)-->(N,FN,out_h,out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        # dout : 反向传播梯度大小(N,FN,out_h,out_w)
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)


class Pooling:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
