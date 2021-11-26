# coding:utf-8
# Loss_TDNet_Q_learning的损失
import tensorflow as tf

class Loss_TDNet_Q_learning():
    def __init__(self):
        pass
    
    def get_loss(self, y_pred, y_true):
        # 论文说的是用的 tf.nn.softmax
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=y_pred,
            labels=y_true
        )
        loss = tf.reduce_mean(loss)
        return loss            