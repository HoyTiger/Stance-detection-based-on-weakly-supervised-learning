# coding:utf-8
# 优化器
import tensorflow as tf

class Optimizer():
    def __init__(self, lr) -> None:
        self.lr = lr
        pass

    def momentum(self, momentum=0.99):
        '''momentum优化器'''
        return tf.compat.v1.train.MomentumOptimizer(self.lr, momentum=momentum)

    def adam(self):
        return tf.compat.v1.train.AdamOptimizer(learning_rate = self.lr)

    def gradient_descent(self):
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr)
