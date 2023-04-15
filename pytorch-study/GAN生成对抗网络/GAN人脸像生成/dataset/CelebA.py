# CelebA数据集

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os,time

class CelebADataset(Dataset):
    def __init__(self, file):
        if not os.path.exists(file):   print("数据集不存在")
        else:
            print("导入数据集")
            fobj = h5py.File(file, "r")
            self.dataset = fobj["img_align_celeba"]

            self.a = [1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self.dataset): raise IndexError()
        img = torch.FloatTensor(np.array(self.dataset[str(index+1)+".jpg"]))

        return img/255.0
    
    def plot_image(self, index):
        image = np.array(self.dataset[str(index)+".jpg"])
        figure, ax = plt.subplots()
        im = ax.imshow(image)
        plt.colorbar(im)
        plt.show()

    def run_images(self):

        # 交互（interactive）模式。即使在脚本中遇到plt.show()，代码还是会继续执行
        plt.ion()
        image = np.random.randint(0, 255, size=(218, 178, 3))
        figure, ax = plt.subplots()
        im = ax.imshow(image)

        for index in range(1, len(self.dataset)+1):
            image = np.array(self.dataset[str(index)+".jpg"])
            # 画图更新
            im.set_data(image)  # update image data
            # draw and flush the figure
            figure.canvas.draw()
            figure.canvas.flush_events()
            # 等待
            time.sleep(1)
        plt.show()

if __name__ == "__main__":

    file = r".//celeba//celeba_aligned_small.h5py"
    CelebA = CelebADataset(file)
    print(len(CelebA))
    # CelebA.plot_image(1)
    # CelebA.run_images()
    for im in CelebA:
        pass

