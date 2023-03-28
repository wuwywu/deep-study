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
import matplotlib.pyplot as plt

# 导入数据集函数
from importDataset import read_dataset_mnist as rdm

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
    def train(self, inputs_list, targets_list):
        '''
            训练网络分为两步：
            1、给定训练样本计算输出
            2、将计算得到的输出与设置的标准输出做对比，得到误差函数指导网络权重更新
        '''
        # 将数据转化为一维列向量
        inputs = np.array(inputs_list).reshape(self.inodes, 1)
        targets = np.array(targets_list).reshape(self.onodes, 1)

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

        # 损失函数的导数loss=1/2*(Y-O)^2
        loss = -(targets-output_outputs)

        # 输出层的delta
        delta_output = loss*output_outputs*(1-output_outputs)
        
        # 隐藏层的delta
        delta_hidden = hidden_outputs*(1-hidden_outputs)*(np.dot(self.who.T, delta_output))
        
        # 更新 -- 隐藏==输出：who
        self.who -= np.dot(delta_output, hidden_outputs.T)*self.lr
       
        # 更新 -- 输入--隐藏：wih
        self.wih -= np.dot(delta_hidden, inputs.T)*self.lr

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
    
    # 初始化权重使----用两种方法设定：1、均匀随机-0.5-0.5；2、正态分布
    def _initWeight(self, choose_way):       
        if choose_way==1:
            wih = np.random.rand(self.hnodes, self.inodes)-0.5
            who = np.random.rand(self.onodes, self.hnodes)-0.5
        elif choose_way==2:
            wih = np.random.normal(0.0, self.hnodes**-0.5, (self.hnodes, self.inodes))
            who = np.random.normal(0.0, self.onodes**-0.5, (self.onodes, self.hnodes))
        else:   print("初始化权重错误")

        return wih, who

    # 使用sigmoid函数 ---- 1/(1+e^-x)，在scipy库中：scipy.special.expit(x)
    # 以后可以更新，可能可以使用生物模型
    def _activation_function(selt, in_act):       
        return scipy.special.expit(in_act)


if __name__=="__main__":

    # 读取数据集
    path = r"./dataset"
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
    epochs = 200

    # 初始化标准对比输出
    targets = np.zeros(output_nodes)+0.01

    # 创建网络
    NN_wy = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
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
            targets[la_num] = 0.01

    # 训练完成
    print("\n  ===训练完成！！！\n")

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
            outputs = NN_wy.query(inputs[i])
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
            outputs = NN_wy.query(inputs[test_num])

            label = np.argmax(outputs)

            image = inputs[test_num].reshape(28,28)

            # 画图
            plt.ion()
            figure, ax = plt.subplots()
            im = ax.imshow(image)
            plt.title('the label_recognition is : {}\nthe label_Correct is : {}'.format(label, labels_correct[test_num]))
            plt.show()

    pass
