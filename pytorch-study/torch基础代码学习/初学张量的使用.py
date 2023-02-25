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
''' 初识PyTorch
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
'''

# 开始使用PyTorch构建神经网络
import torch
import torch.nn as nn
import sys
import numpy as np

sys.path.append("../深度学习原理知识学习")
# 导入数据集函数
from importDataset import read_dataset_mnist as rdm

# 1、继承torch.nn模块中的很多功能，如：自动构建计算图，查看权重，在训练期间更新权重，等

class Classifier(nn.Module):

    # Sequential的作用就是连接模型的功能，使代码看起来更加整洁
    def __init__(self, inputnodes, hiddenodes, outputnodes, learning_rate):
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(inputnodes, hiddenodes),
            nn.Sigmoid(),
            nn.Linear(hiddenodes, outputnodes),
            nn.Sigmoid()
        )

        # 创建损失函数
        '''
            the input :math:`x` and target :math:`y`
            .. math::
                \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
                l_n = \left( x_n - y_n \right)^2,
        '''
        self.loss_fun = nn.MSELoss()

        # 创建优化器，使用简单的梯度下降
        '''
            stochastic gradient descent, SGD
            使用self.parameters()访问所有可以学习的参数---=linear中包含权重和偏差
        '''
        self.optimiser = torch.optim.SGD(self.parameters(), lr=learning_rate)


    def forward(self, x):
        # 运行模型---传入输入数据
        return self.model(x)
    
    def train(self, inputs_list, targets_list):
        # train the neural network
        '''
            训练网络分为两步：
            1、给定训练样本计算输出
            2、将计算得到的输出与设置的标准输出做对比，得到误差函数指导网络权重更新
        '''
        # 将数据转化为一位列向量
        inputs = np.array(inputs_list).reshape(self.inodes, 1)
        targets = np.array(targets_list).reshape(self.onodes, 1)

        # 计算网络的输出值
        outputs = self.forward(inputs)

        # 计算损失值
        loss = self.loss_fun(outputs, targets)

        # =================== 使用损失值更新权重 ===================
        # 梯度归零，方向传播，更新权重值
        self.optimiser.zero_grad()  # 将图中的梯度全部归零
        loss.backward()             # 计算网络中的梯度
        self.optimiser.step()       # 更新网络中可学习的参数
        pass

if __name__=="__main__":

    # 读取数据集
    path = r"../深度学习原理知识学习/dataset"
    filename_images = "train-images-idx3-ubyte"
    filename_labels = "train-labels-idx1-ubyte" 
    read_dataset_train = rdm(path, filename_images, filename_labels)
    inputs, labels = read_dataset_train.turn_dataset()

    # 给定输入层，隐藏层，输出层的节点数量
    input_nodes = read_dataset_train.rows*read_dataset_train.cols
    hidden_nodes = 200
    output_nodes = 10

    # 给定学习率
    learning_rate = 0.01

    NN_wy = Classifier(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # print(NN_wy.parameters())
    pass