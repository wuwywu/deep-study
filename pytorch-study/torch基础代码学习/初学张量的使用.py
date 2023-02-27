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
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../深度学习原理知识学习")
# 导入数据集函数
from importDataset import read_dataset_mnist as rdm

# GPU usage #########################################
# dtype = torch.FloatTensor # uncomment if you are using CPU
dtype = torch.cuda.FloatTensor # uncomment if you are using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1、继承torch.nn模块中的很多功能，如：自动构建计算图，查看权重，在训练期间更新权重，等

class Classifier(nn.Module):

    # Sequential的作用就是连接模型的功能，使代码看起来更加整洁
    def __init__(self, inputnodes, hiddenodes, outputnodes, learning_rate):
        super().__init__()

        # 设置输入层，隐藏层，输出层的节点数量
        self.inodes = inputnodes
        self.hnodes = hiddenodes
        self.onodes = outputnodes

        # 记录训练进展的计数器和列表
        self.counter = 0
        self.progress = []

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(inputnodes, hiddenodes),
            nn.Sigmoid(),
            nn.Linear(hiddenodes, outputnodes),
            nn.Sigmoid()
        ).type(dtype).to(device)

        # 创建损失函数
        '''
            the input :math:`x` and target :math:`y`
            .. math::
                \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
                l_n = \left( x_n - y_n \right)^2,

            reduction=“none”, "mean", "sum"
        '''
        self.loss_fun = nn.MSELoss()

        # 创建优化器，使用简单的梯度下降
        '''
            stochastic gradient descent, SGD
            使用self.parameters()访问所有可以学习的参数---=linear中包含权重和偏差
        '''
        self.optimiser = torch.optim.SGD(self.parameters(), lr=learning_rate)
    
    def train(self, inputs_list, targets_list):
        # train the neural network
        '''
            训练网络分为两步：
            1、给定训练样本计算输出
            2、将计算得到的输出与设置的标准输出做对比，得到误差函数指导网络权重更新
        '''
        # 将数据转化为一维横向量
        inputs = torch.from_numpy(np.array(inputs_list).reshape(1, self.inodes)).type(dtype).to(device)
        targets = torch.from_numpy(np.array(targets_list).reshape(1, self.onodes)).type(dtype).to(device)
        # print(type(inputs))

        # 计算网络的输出值
        outputs = self.forward(inputs)

        # 计算损失值
        loss = self.loss_fun(outputs, targets)
        # print("loss={}".format(loss.item()))

        # =================== 使用损失值更新权重 ===================
        # 梯度归零，方向传播，更新权重值
        self.optimiser.zero_grad()  # 将图中的梯度全部归零
        loss.backward()             # 计算网络中的梯度
        self.optimiser.step()       # 更新网络中可学习的参数

        # 训练次数加一
        self.counter += 1

        # 记录损失函数
        if self.counter%100==0:
            self.progress.append(loss.item())
        pass

    def forward(self, x):
        # 运行模型---传入输入数据
        return self.model(x)
    
    # query the neural netwok
    def query(self, inputs_list):
        # 将数据转化为一维横向量
        inputs = torch.from_numpy(np.array(inputs_list).reshape(1, self.inodes)).type(dtype).to(device)

        # 计算网络的输出值
        outputs = self.forward(inputs)
        
        return outputs

    def plot_progress(self):
        # 画出损失函数
        # print(self.progress)
        plt.ion()
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0,0.25), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.125,0.25))
        plt.show()
        pass

if __name__=="__main__":

    # 读取数据集
    path = r"../深度学习原理知识学习/dataset"
    filename_images = "train-images-idx3-ubyte"
    filename_labels = "train-labels-idx1-ubyte" 
    read_dataset_train = rdm(path, filename_images, filename_labels)
    inputs, labels = read_dataset_train.turn_dataset()
    # 将输入0~255变为0~1
    inputs = (np.asfarray(inputs))/255*0.99+0.01

    # 给定输入层，隐藏层，输出层的节点数量
    input_nodes = read_dataset_train.rows*read_dataset_train.cols
    hidden_nodes = 200
    output_nodes = 10

    # 给定学习率
    learning_rate = 0.01

    # 给定数据集训练次数
    epochs = 1000

    # 初始化标准对比输出
    targets = np.zeros(output_nodes)+0.01

    NN_wy = Classifier(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # print(NN_wy.parameters())
    print("=======创建网络成功=======\n")
    for e in range(epochs):
        if e%1==0:    print("\r", "学习进度：{:.3f} %".format(e*100/epochs), end='', flush=True)
        # 所有数据集：read_dataset_train.images_num
        for i in range(read_dataset_train.images_num):
            # 生成标准对比输出
            la_num = int(labels[i])
            targets[la_num] += 0.99

            # 训练网络
            NN_wy.train(inputs[i], targets)

            # 初始化标准对比输出
            targets[la_num] = 0

    # 训练完成
    print("\n  ===训练完成！！！\n")

    # 画出损失值
    NN_wy.plot_progress()

    #==================== 测试训练效果 ======================
    filename_images = "t10k-images-idx3-ubyte"
    filename_labels = "t10k-labels-idx1-ubyte" 
    read_dataset_test = rdm(path, filename_images, filename_labels)
    inputs, labels_correct = read_dataset_test.turn_dataset()
    # 将输入0~255变为0~1
    inputs = (np.asfarray(inputs))/255*0.99+0.01

    # 保存错误的序号，识别标签，以及正确标签
    file_Test = r'./test_error_correct.txt'
    error_num = 0
    with open(file_Test, "w") as f:   
        f.write("样本数：{},\t训练轮数：{}\n".format(read_dataset_test.labels_num, epochs))
        f.write("序号\tlabel\tlabels_correct\n")
        for i in range(read_dataset_test.labels_num):
            outputs = NN_wy.query(inputs[i]).cpu().detach().numpy()
            label = np.argmax(outputs)
            if not(int(label) == int(labels_correct[i])):
                error_num += 1
                # 写入错误的序号，识别标签，以及正确标签
                f.write("{}\t{}\t{}\n".format(i+1, label, labels_correct[i]))

    print("识别错误的个数：{}；测试学习的成功率为：{}\n".format(error_num, (read_dataset_test.labels_num-error_num)/read_dataset_test.labels_num))

    while_judge1 = True
    while_judge2 = True
    print("进入选择测试训练效果\n")
    while while_judge1:
        # 输入数值
        while while_judge2:
            test_num = input("测试请输入数字{}~{}(输入e退出): ".format(1, read_dataset_test.labels_num))
            # 是e就退出
            if test_num=="e":   
                while_judge1 = False
                break
                
            if not test_num.isdigit():
                print("请输入数字!!!\n")
                continue

            test_num = int(test_num)-1

            if not(test_num>=0 and test_num<=read_dataset_test.labels_num-1):
                print("输入范围错误，请重输!!!\n")
            else: break
        
        # 测试
        if not(test_num=="e"):
            outputs = NN_wy.query(inputs[test_num]).cpu().detach().numpy()

            label = np.argmax(outputs)

            image = inputs[test_num].reshape(28,28)

            # 画图
            plt.ion()
            figure, ax = plt.subplots()
            im = ax.imshow(image)
            plt.title('the label_recognition is : {}\nthe label_Correct is : {}'.format(label, labels_correct[test_num]))
            plt.show()

    pass
