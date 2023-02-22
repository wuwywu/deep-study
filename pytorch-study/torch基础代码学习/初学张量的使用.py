# 初次学习PyTorch
'''
目标：使用PyTorch创建深度神经网络
学习使用PyTorch：
    1、张量的使用
    2、网络的构建
    3、使用CUDA---显卡加速运算
    4、验证网络的正确性
    5、改bug
    6、改良网络
    7、思考怎么加入神经元模型，结合以前的研究，并更进一步
'''
# 导入torch
import torch

# 创建张量
# 通过requires_grad告诉PyTorch我们希望得到一个关于x的梯度
# x = torch.tensor(3.5, requires_grad=True) 
a = torch.tensor(2.0, requires_grad=True)  
b = torch.tensor(1.0, requires_grad=True)

# 创建一个张量的函数关系(计算关系图)---可以动态的创建计算关系图
# y = (x-1)*(x-2)*(x-3)
# y = x**2
# z = 2*y+3
x = 2*a+3*b
y = 5*a**2+3*b**3
z = 2*x+3*y

# 计算梯度
z.backward()

# 输出x=3.5时的梯度
print(a.grad, b.grad)

