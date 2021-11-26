# coding:utf-8
# Loss_SRNet_Q_learning的损失
import tensorflow as tf
class Loss_SRNet_Q_learning():
    def __init__(self) -> None:
        
        pass

    def get_loss(self, y_pred, y_true):
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
        return loss