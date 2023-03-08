# 从基础学起

# 导入包
import torch
import numpy as np

# 1、数据类型
'''
    int                     IntTensor
    float                   FloatTensor
    int array               IntTensor[d1,d2,...]
    float array             FloatTensor[d1,d2,...]
    string                  使用编码的方式表示string类型
'''

dtype_cpu = torch.FloatTensor # uncomment if you are using CPU
dtype_gpu = torch.cuda.FloatTensor # uncomment if you are using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# a = torch.randn(2,3)
# print(a.type())     # torch.FloatTensor

# 判断数据类型
# print(isinstance(a, torch.FloatTensor))    # Trut

# 默认是torch.cuda.FloatTensor类型32位浮点类型(gpu)
# a = a.cuda()
# print(a.type()) # torch.cuda.FloatTensor

# b = torch.randn(2,3).type(dtype_gpu)
# print(b.type()) # torch.cuda.FloatTensor

# 标量
# b1 = torch.tensor(1.)
# print(b1.type())    # torch.FloatTensor

# b2 = torch.tensor(1)
# print(b2.type())    # torch.LongTensor 64位整型

# shape与size()用法相同
# print(a)
# print(a.shape)
# print(a.size())

# 向量
# x1 = torch.tensor([1.2, 2.3])
# print(x1)    # torch.FloatTensor
# x2 = torch.FloatTensor(2)   # 随机初始化维度
# print(x2)    # torch.FloatTensor
# x3_np = np.ones(2)
# print(type(x3_np))  # numpy.ndarray
# x3 = torch.from_numpy(x3_np) # .type(dtype_gpu).to(device)
# print(x3.type())
a = torch.zeros(100,100).type(dtype_cpu)
# print(a)
while True:
    a += 1
#     pass







