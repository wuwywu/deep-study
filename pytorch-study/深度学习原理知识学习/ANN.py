# wuyong@ccnu
# 学习人工神经网络的运行原理
# 学习原理----学习使用PyTorch
# 从小处入手，然后让程序慢慢长大
'''
框架代码：
1、初始化函数----设定输入层节点、隐藏层节点和输出层节点的数量
2、训练----学习给定训练集样本后、优化权重
3、查询----给定输入，从输出节点给出答案
'''
import numpy as np


# neural network class
class neuralNetwork:

    # initialise the neural network
    def __init__(self,inputnodes,hiddenodes,outputnodes,learningrate):
        # 设置输入层，隐藏层，输出层的节点数量
        self.inodes = inputnodes
        self.hnodes = hiddenodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        # 设置初始权重
        
        pass

    # train the neural network
    def train(self):
        pass
    
    # query the neural netwok
    def query(self):
        pass



if __name__=="__main__":

    # 给定输入层，隐藏层，输出层的节点数量
    input_nodes = 28
    hidden_nodes = 28
    output_nodes = 10

    # 给定学习率
    learning_rate = 0.001

    # 创建网络
    NN_wy = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    # print(NN_wy.inodes, NN_wy.lr)

    pass
