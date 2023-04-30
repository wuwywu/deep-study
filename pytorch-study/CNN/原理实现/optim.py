'''
各类优化器：
1、SGD : 随机梯度下降法（Stochastic Gradient Descent）
2、Momentum SGD : SGD的基础上加上动量

'''

import numpy as np

class SGD:
    """随机梯度下降法（Stochastic Gradient Descent）"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        '''
        :params : 包含了关键词和值的可学习参数（权重和偏置量）
        grads : 计算的可学习参数的梯度
        '''
        for key in params.key:
            params[key] -= self.lr*grads[key]


class Momentum:
    """Momentum SGD"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None               # 初始化速度（None空对象）

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

