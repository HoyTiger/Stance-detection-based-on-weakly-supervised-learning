# coding:utf-8
# Bi_GRU的实现
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)

class TDNet_Q_learning():
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        # self.ls_hidden_layer = [128, 256, 512, 1024, 1024]
        # self.ls_hidden_layer = [128, 256, 512]
        self.ls_hidden_layer = [128, 256, 256]
        # self.ls_hidden_layer = [128, 256]

        pass

    def get_parameter_num(self):
        num = 0
        for i in range(len(self.ls_hidden_layer)-1):
            num += self.ls_hidden_layer[i] * self.ls_hidden_layer[i+1]
        # 加上第一层
        num += self.ls_hidden_layer[0]
        # 卷积核长度为3
        num *= 3
        # 加上最后一层
        num += self.ls_hidden_layer[-1]*2 * self.num_classes + self.num_classes
        return num

    
    def get_weight(self, shape,regularizer=0.0005):
        w = tf.Variable(tf.random.truncated_normal(shape,stddev = 0.1), name="Conv")
        #加入正则化
        if regularizer != None:
            tf.compat.v1.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

    #偏置
    def get_bias(self, shape, stddev=0.1):
        b = tf.Variable(tf.random.truncated_normal(shape,stddev=stddev), name="bias")   # , seed=2222
        return b

    def RNN_Bi_GRU(self, inputs):

        ls_bi_rnn_fw_cell_cell = [tf.nn.rnn_cell.GRUCell(num_hidden) for num_hidden in self.ls_hidden_layer]
        ls_bi_rnn_back_cell_cell = [tf.nn.rnn_cell.GRUCell(num_hidden) for num_hidden in self.ls_hidden_layer]
        
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(ls_bi_rnn_fw_cell_cell)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(ls_bi_rnn_back_cell_cell)

        outputs, _=tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32) 
        rnn_output=tf.concat(outputs, 2)

        return rnn_output

    def forward(self, inputs, actions):
        '''
        得到这个环境下估计的每一种选择的reward
        actions:[batch, 2]
        '''
        with tf.compat.v1.variable_scope("td_net"):
            rnn_outputs = self.RNN_Bi_GRU(inputs)

            in_channels = self.ls_hidden_layer[-1]*2
            weights = self.get_weight([in_channels, self.num_classes])
            bias = self.get_bias([self.num_classes])

            # outputs = tf.nn.softmax(tf.matmul(rnn_outputs[:, -1, :], weights)+bias, 1)
            outputs = tf.matmul(rnn_outputs[:, -1, :], weights)+bias

            reward_action = tf.reduce_mean(actions * tf.clip_by_value(-1*tf.math.log(tf.nn.softmax(outputs, axis=-1)), 0, 10) )
        return outputs, reward_action
        

