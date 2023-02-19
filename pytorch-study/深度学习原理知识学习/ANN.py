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
import scipy

# neural network class
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddenodes, outputnodes, learningrate):
        # 设置输入层，隐藏层，输出层的节点数量
        self.inodes = inputnodes
        self.hnodes = hiddenodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        # 设置初始权重（使用两种方法设定：1、均匀随机-0.5-0.5；2、正态分布）
        # 输入--隐藏：wih；隐藏==输出：who
        self.wih, self.who = self._initWeight(1)
        
        pass

    # train the neural network
    def train(self):
        pass
    
    # query the neural netwok
    def query(self, inputs_list):
        # 将数据转化为一位列向量
        inputs = np.array(inputs_list).reshape(self.inodes, 1)
        
        #=================== 第二层计算 ===================
        # 计算隐藏层输入
        hidden_inputs = np.dot(self.wih, inputs)

        # 计算隐藏层输出（经过激活函数）
        hidden_outputs = self._activation_function(hidden_inputs)

        #=================== 第三层计算 ===================
        # 计算输出层输入
        output_inputs = np.dot(self.who, hidden_outputs)

        # 计算输出层输出（经过激活函数）
        output_outputs = self._activation_function(output_inputs)
        
        return output_outputs

    def _initWeight(self, choose_way):
        # 初始化权重
        if choose_way==1:
            wih = np.random.rand(self.hnodes, self.inodes)-0.5
            who = np.random.rand(self.onodes, self.hnodes)-0.5
        elif choose_way==2:
            wih = np.random.normal(0.0, self.hnodes**-0.5, (self.hnodes, self.inodes))
            who = np.random.normal(0.0, self.onodes**-0.5, (self.hnodes, self.inodes))
        else:   print("初始化权重错误")

        return wih, who
    
    def _activation_function(selt, in_act):
        # 使用sigmoid函数 ---- 1/(1+e^-x)，在scipy库中：scipy.special.expit(x)
        # 以后可以更新，可能可以使用生物模型
        return scipy.special.expit(in_act)


if __name__=="__main__":

    # 给定输入层，隐藏层，输出层的节点数量
    input_nodes = 9
    hidden_nodes = 3
    output_nodes = 3

    # 给定学习率
    learning_rate = 0.001

    # 创建网络
    NN_wy = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


    # a = np.random.rand(3, 3)
    # print(NN_wy.query(a))

    # print(NN_wy.who)

    pass
