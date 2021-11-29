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
from net.SRNet_Q_learning import SRNet_Q_learning
from net.TDNet_Q_learning import TDNet_Q_learning
from dataLoader.Data_loader import Data_loader
from loss.Loss_TDNet_Q_learning import Loss_TDNet_Q_learning
from loss.Loss_SRNet_Q_learning import Loss_SRNet_Q_learning
from optimizer.Optimizer import Optimizer
from learning_rate.Learning_rate import Learning_rate
import numpy as np
import os.path as osp
from tqdm import tqdm
import bitermplus as btm
from utils import tools
import gc

size = config.size
batch_size = config.batch_size
num_classes = 2 # 做二分类
action_num = 2  # 两种动作
train_data_file = config.train_data_file
test_data_file = "./Data/testdata-taskB-all-annotations.txt"
lr_base = config.lr_base
lr_min = config.lr_min
save_per_epoch = config.save_per_epoch
warm_epoch = config.warm_epoch
model_dir = "./model_Q_learning_2classes"
model_name = "model"
temp_model_dir = osp.join(model_dir, model_name+"-temp")
best_model_dir = osp.join(model_dir, model_name+"-best")
log_dir = "./log_Q_learning_2classes"
log_name = "log.txt"

# add a log message
def add_log(content):
    if not osp.isdir(log_dir):
        os.makedirs(log_dir)
        add_log("message:create folder '{}'".format(log_dir))
    log_file = osp.join(log_dir, log_name)
    print(content)
    tools.write_file(log_file, content, True)
    gc.collect()
    return

def do_actions(pre_actions, ls_batch_x, ls_batch_y, del_max=(batch_size-4)):
    '''执行pre_actions做动作，删除对应的数据，0删除1保留'''
    # 如果删的太多了就保留一部分
    while(len(pre_actions) - np.sum(pre_actions) > del_max):
        a = len(pre_actions)
        b = np.sum(pre_actions)
        # 随机保留
        ls_del = np.where(pre_actions==0)[0]
        ls_save_index = np.random.choice(ls_del, len(pre_actions)-del_max - int(np.sum(pre_actions)), replace=False)
        for index in ls_save_index:
            pre_actions[index][0] = 1
        pass
    ls_del_index = []
    ls_save_index = []  # 保留的部分
    for index in range(len(pre_actions)):
        if pre_actions[index] == 0:
            ls_del_index.append(index)
        else:
            ls_save_index.append(index)

    # do actions
    ls_batch_x_d, ls_batch_y_d = np.delete(ls_batch_x, ls_del_index, axis=0), np.delete(ls_batch_y, ls_del_index, axis=0)
    return ls_batch_x_d, ls_batch_y_d, ls_save_index, pre_actions

def make_SRNet_label(value, value1, ls_save_index, gamma=0.99):
    '''
    通过value和value1制作SRNet的标签
    args:
        value: TDNet在Data下计算的value
        value1: TDNet在Data'下计算的value
        ls_save_index: Data中保留了哪些index的数据
        gamma: 折扣回报率
    '''
    for i, index in enumerate(ls_save_index):
        value[index] = value[index] + value1[i]*gamma
    return value

def backward():
    data = Data_loader(train_data_file, num_classes, size, batch_size)
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





    global_step = tf.Variable(0, trainable=False, name="global_step")
    # global_step = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name="global_step")

    # srnet和tdnet的输入都是data
    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, size, 1], name="inputs")
    
    ''' ######### sr net ######### '''
    sr_y_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="sr_y_true")
    sr_net = SRNet_Q_learning()
    # 应该删除那些标签，保留哪些标签
    sr_pred_actions = sr_net.forward(inputs=inputs)
    print("sr net 参数:{}k".format(sr_net.get_parameter_num()/1000.0))

    ''' ######### td net ######### '''
    # td_net的标签是来自于Data
    td_net_y_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_classes], name="td_net_y_true")
    td_net = TDNet_Q_learning(num_classes=num_classes)
    # 在这样的环境下，每种动作得到的最大回报
    td_pred_outputs, td_pred_rewards = td_net.forward(inputs=inputs, actions=sr_pred_actions)
    print("td net 参数:{}k".format(td_net.get_parameter_num()/1000.0))

    ''' ######### loss td net ######### '''
    # td_net的更新使用交叉熵
    loss_td_net = Loss_TDNet_Q_learning().get_loss(y_pred=td_pred_outputs, y_true=td_net_y_true)
    lr_td_net = Learning_rate(lr_base, lr_min=lr_min).warmup_Cosine_lr(
                                                                        global_step=global_step,
                                                                        total_step=data.steps_per_epoch * 500,
                                                                        warmup_step=warm_epoch*data.steps_per_epoch
                                                                    )
    optimizer_td_net = Optimizer(lr_td_net).adam()
    update_ops_td_net = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_td_net):
        gvs = optimizer_td_net.compute_gradients(loss_td_net)
        clip_grad_var = [gv if gv[0] is None else[tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        # train_op_td_net = optimizer_td_net.apply_gradients(clip_grad_var, global_step=global_step)
        train_op_td_net = optimizer_td_net.apply_gradients(clip_grad_var)

    ''' ######### loss sr net ######### '''
    loss_sr_net = Loss_SRNet_Q_learning().get_loss(y_true=sr_y_true, y_pred=td_pred_rewards)
    lr_sr_net = Learning_rate(lr_base, lr_min=lr_min).warmup_Cosine_lr(
                                                                        global_step=global_step,
                                                                        total_step=data.steps_per_epoch * 500,
                                                                        warmup_step=warm_epoch*data.steps_per_epoch
                                                                    )
    optimizer_sr_net = Optimizer(lr_sr_net).adam()
    update_ops_sr_net = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_sr_net):
        gvs = optimizer_sr_net.compute_gradients(loss_sr_net)
        clip_grad_var = [gv if gv[0] is None else[tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op_sr_net = optimizer_sr_net.apply_gradients(clip_grad_var, global_step=global_step)


    saver = tf.compat.v1.train.Saver(max_to_keep=250)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # sess.run(iterator.initializer)

        ckpt = tf.compat.v1.train.get_checkpoint_state(temp_model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("message: load ckpt model, global_step=" + str(step))
        else:
            print("message:can not fint ckpt model")
            step = 0

        acc_max = 0
        while True:
            td_loss_total = 0
            sr_loss_total = 0
            num_del = 0
            for _ in tqdm(range(data.steps_per_epoch)):
                x_batch, y_batch = next(data)

                # 1. 当前环境下应该做什么action并且应该得到怎样的环境结果及其reward，以及老的决策
                sr_actions, outputs = sess.run([sr_pred_actions, td_pred_outputs], 
                                                                            feed_dict={inputs:x_batch})

                # 2. 更新环境
                new_data, new_label, _, sr_actions = do_actions(sr_actions, x_batch, y_batch)
                num_del += (batch_size - len(new_data))
                _, td_loss = sess.run([train_op_td_net, loss_td_net], feed_dict={inputs:new_data, td_net_y_true:new_label})
                td_loss_total += td_loss

                # 3. 环境更新后的reward
                [new_reward] = sess.run([td_pred_rewards], feed_dict={inputs:x_batch,sr_pred_actions:sr_actions})

                # 4. 更新agent
                _, sr_loss, lr, step = sess.run([train_op_sr_net, loss_sr_net, lr_sr_net, global_step], feed_dict={
                                                            inputs:x_batch, 
                                                            td_pred_outputs:outputs,
                                                            sr_y_true:new_reward})
                sr_loss_total += sr_loss

                x_batch = None
                y_batch = None
                new_data = None
                new_label = None

                pass

            curr_epoch = np.math.floor(step/data.steps_per_epoch)
            add_log("message: epoch={}, lr:{}, avg_td_loss:{}, avg_sr_loss:{}, avg_del:{}".format(
                                                                                curr_epoch, lr, 
                                                                                td_loss_total/data.steps_per_epoch, 
                                                                                sr_loss_total/data.steps_per_epoch,
                                                                                num_del/data.steps_per_epoch)
                                                                            )

            if curr_epoch % save_per_epoch == 0:
                # save ckpt model
                print("message: save ckpt model, step:{}".format(step))
                saver.save(sess, osp.join(model_dir, model_name), global_step=step)

            # save temp_model
            print("message:saving the temp model, epoch:{}, step:{}".format(curr_epoch, step))
            saver.save(sess, osp.join(temp_model_dir, model_name+"-temp"))

            # 测试
            [output] = sess.run([td_pred_outputs],feed_dict={inputs:test_x})
            curr_acc, _ = tools.f1(y_pred=output, y_true=test_y, num_classes=num_classes)
            curr_acc *= 100
            print("f1 {}%".format(curr_acc))
            if curr_acc > acc_max:
                acc_max = curr_acc
                # save best temp_model
                add_log("message:saving the best model, epoch:{}, step:{}, acc:{}%".format(curr_epoch, step, curr_acc))
                saver.save(sess, osp.join(best_model_dir, model_name+"-{}".format(curr_acc)))
    return

def main():
    if not osp.isdir(model_dir):
        os.makedirs(model_dir)
        print("建立文件夹 {}".format(model_dir))

    backward()
    return 

if __name__ == "__main__":
    main()