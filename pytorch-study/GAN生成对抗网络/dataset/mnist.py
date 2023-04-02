# wuyong@ccnu
# 手写数字数据集下载网址MNIST：http://yann.lecun.com/exdb/mnist/index.html
# train-images-idx3-ubyte.gz: 60000个训练图片
# train-labels-idx1-ubyte.gz: 60000个训练标签
# t10k-images-idx3-ubyte.gz: 10000个测试图片
# t10k-labels-idx1-ubyte.gz: 10000个测试标签
'''
搭配 torch.utils.data.DataLoader
使用PyTorch中的 torch.utils.data.Dataset 
'''
import os,time
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class MnistDataset(Dataset):

    def __init__(self, filename1, filename2):
        self._read_dataset_fromPath(filename1, filename2)
    
    def _read_dataset_fromPath(self, filename1, filename2):
        if os.path.exists(filename1):
            with open(filename1, 'rb') as imgpath:
                print("dataset_images is existing")
                self.images_magic, self.images_num, self.rows, self.cols = struct.unpack('>IIII', imgpath.read(16))
                self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(self.images_num, self.rows * self.cols)
        else:   print("dataset_images is inexisting")

        if os.path.exists(filename2):
            print("dataset_labels is existing")
            with open(filename2, 'rb') as lbpath:
                self.labels_magic, self.labels_num = struct.unpack('>II', lbpath.read(8))
                self.labels = np.fromfile(lbpath, dtype=np.uint8)
        else:   print("dataset_labels is inexisting")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 目标图像 (标签)
        label = self.labels[index]
        target = torch.zeros((10))
        target[label] = 1.0

        # 图像数据, 取值范围是0~255，标准化为0.01~1
        image_values = (torch.FloatTensor(self.images[index]))/255*0.99+0.01

        return label, image_values, target

    def plot_image(self, choose_num):
        label = self.labels[choose_num]
        image = self.images[choose_num].reshape(28,28)
        # 交互（interactive）模式。即使在脚本中遇到plt.show()，代码还是会继续执行
        # plt.ion()
        figure, ax = plt.subplots()
        im = ax.imshow(image)
        plt.colorbar(im)
        plt.title('the label is : {}'.format(label))
        plt.show()
        # time.sleep(100)

    def run_images(self):

        # 交互（interactive）模式。即使在脚本中遇到plt.show()，代码还是会继续执行
        plt.ion()
        label = ""
        image = np.random.randint(0, 255, size=(self.rows, self.cols))
        figure, ax = plt.subplots()
        im = ax.imshow(image)
        # plt.colorbar(im)
        plt.title('the label is : {}'.format(label))
        # plt.show()

        for choose_num in range(0, self.images_num):
            # print(choose_num)
            label = self.labels[choose_num]
            image = self.images[choose_num].reshape(28,28)
            # 画图更新
            im.set_data(image)  # update image data
            # draw and flush the figure
            plt.title('the label is : {}'.format(label)) # 图像题目
            figure.canvas.draw()
            figure.canvas.flush_events()
            # 等待
            time.sleep(1)
        plt.show()

    def debug(self):
        print('labels_magic is {} \n'.format(self.labels_magic),
        'labels_num is {} \n'.format(self.labels_num),
        'labels is {} \n'.format(self.labels))

        print('images_magic is {} \n'.format(self.images_magic),
        'images_num is {} \n'.format(self.images_num),
        'rows is {} \n'.format(self.rows),
        'cols is {} \n'.format(self.cols),
        'images is {} \n'.format(self.images))

if __name__=="__main__":
    # path = r"./dataset"
    filename_images = "train-images-idx3-ubyte"
    filename_labels = "train-labels-idx1-ubyte"
    # filename_images = path+"/"+filename_images
    # filename_labels = path+"/"+filename_labels

    Mnist = MnistDataset(filename_images, filename_labels)
    # Mnist.plot_image(1)
    # index = 3
    # label, image_values, target = Mnist[index]
    
    pass
