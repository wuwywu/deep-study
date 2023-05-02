# wuyong@ccnu
'''
网络结构如下所示
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax

1、数据批量化
2、建立CNN网络
'''
from mnist import MnistDataset as MD
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 导入手写数据集(保留数据形状为(N,C,H,W))
md = MD(flatten=False)

