# coding:utf-8
# 训练价值学习
import os

from tqdm.std import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   #设置为1屏蔽一般信息，2屏蔽一般和警告，3屏蔽所有输出

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import config
import tensorflow as tf
from net.TDNet_Q_learning import TDNet_Q_learning
from dataLoader.Data_loader import Data_loader
from tqdm import tqdm
import bitermplus as btm
import numpy as np
from utils import tools

size = config.size
batch_size = 64
num_classes = 2
train_data_file = config.train_data_file
test_data_file = "./Data/testdata-taskB-all-annotations.txt"
model_dir = "model_Q_learning_2classes/model-best/{}"
# model_name = "model-3200"
# model_name = "model-12800"
# model_name = "model-6400"
model_name = "model-58.758058488274315"

def forward():
    data = Data_loader(train_data_file, num_classes, size, batch_size)
    data.tongji()
    # #### 测试数据 ####
    data_test = Data_loader(test_data_file, num_classes, size, batch_size)
    ls_vectorized_tweet = btm.get_vectorized_docs(data_test.ls_tweet, data.map_word2id)
    '''把单词向量padding 0'''
    ls_padding_vector_tweet = []
    for line in tqdm(ls_vectorized_tweet):
        curr = [0] * size
        curr[0: min(size, len(line))] = line
        ls_padding_vector_tweet.append(curr)
    # 标签
    ls_labels = data_test.ls_labels
    # 全部测试数据
    test_x = []
    test_y = []
    for index in range(len(data_test)):
        x = ls_padding_vector_tweet[index]
        y = ls_labels[index]
        test_x.append(x)
        test_y.append(y)
    test_x = np.asarray(test_x).astype(np.float32)
    test_x = np.expand_dims(test_x, -1)
    test_y = np.asarray(test_y).astype(np.float32)


    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, size, 1], name="inputs")
    td_net = TDNet_Q_learning(num_classes=num_classes)
    pred_outputs, _ = td_net.forward(inputs=inputs, actions=inputs)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, model_dir.format(model_name))
                
        [output] = sess.run([pred_outputs],feed_dict={inputs:test_x})
        curr_acc, _ = tools.f1(y_pred=output, y_true=test_y, num_classes=num_classes)
        print("f1 {}%".format(curr_acc*100.0))

    return

if __name__ == "__main__":
    forward()