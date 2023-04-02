# wuyong@ccnu
# 学习GAN
# 学习原理----学习使用PyTorch
# 从小处入手，然后让程序慢慢长大
'''
框架代码：
1、初始化函数----设定生成器和鉴别器
2、训练----学习给定训练集样本后、优化权重
3、查询，生成图像
'''
import os,sys
sys.path.append("./dataset")
from mnist import MnistDataset

path = r"./dataset"

filename_images = "train-images-idx3-ubyte"
filename_labels = "train-labels-idx1-ubyte"
filename_images = path+"/"+filename_images
filename_labels = path+"/"+filename_labels


Mnist_train = MnistDataset(filename_images, filename_labels)
index = 3
label, image_values, target = Mnist_train[index]
print("length_train =", len(Mnist_train))

filename_images = "t10k-images-idx3-ubyte"
filename_labels = "t10k-labels-idx1-ubyte"
filename_images = path+"/"+filename_images
filename_labels = path+"/"+filename_labels
Mnist_test = MnistDataset(filename_images, filename_labels)
label, image_values, target = Mnist_test[index]
print("length_test =", len(Mnist_test))

# print(target)








