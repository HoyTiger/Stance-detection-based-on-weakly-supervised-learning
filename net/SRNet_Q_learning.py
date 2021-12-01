# coding:utf-8
# SRNet_Q_learning
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)

class SRNet_Q_learning():
    '''判断是不是应该删除这一个标签，0删除1保留'''
    def __init__(self) -> None:
        self.num_action = 2
        self.ls_hidden_layer = [128, 256, 256]
        # self.ls_hidden_layer = [128, 256, 512]
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
        num += self.ls_hidden_layer[-1]*2 * self.num_action + self.num_action
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
        "bi gru网络"

        ls_bi_rnn_fw_cell_cell = [tf.nn.rnn_cell.GRUCell(num_hidden) for num_hidden in self.ls_hidden_layer]
        ls_bi_rnn_back_cell_cell = [tf.nn.rnn_cell.GRUCell(num_hidden) for num_hidden in self.ls_hidden_layer]
        
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(ls_bi_rnn_fw_cell_cell)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(ls_bi_rnn_back_cell_cell)

        # 输出(outputs, output_states),outputs为(output_fw, output_bw),states为(output_state_fw, output_state_bw)
        outputs, states=tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32) 
        rnn_output=tf.concat(outputs, 2)

        return rnn_output

    def forward(self, inputs):
        '''在inputs情况下执行actions估计得到的value'''
        rnn_outputs = self.RNN_Bi_GRU(inputs)

        in_channels = self.ls_hidden_layer[-1]*2 
        weights = self.get_weight([in_channels, self.num_action])
        bias = self.get_bias([self.num_action])

        # outputs = tf.nn.softmax(tf.matmul(rnn_outputs[:, -1, :], weights)+bias, 1)
        actions_del_labels = tf.matmul(rnn_outputs[:, -1, :], weights)+bias

        actions = tf.cast( tf.expand_dims(tf.argmax(actions_del_labels, axis=-1), -1), tf.float32)
        # actions = tf.cast(tf.argmax(actions_del_labels, axis=-1), tf.float32)
        return actions


