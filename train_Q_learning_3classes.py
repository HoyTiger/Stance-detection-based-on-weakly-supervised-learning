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
from utils import tools

size = config.size
batch_size = config.batch_size
num_classes = 3 # 做二分类
action_num = 2  # 两种动作
train_data_file = config.train_data_file
test_data_file = "./Data/testdata-taskB-all-annotations.txt"
lr_base = config.lr_base
lr_min = config.lr_min
save_per_epoch = config.save_per_epoch
warm_epoch = config.warm_epoch
model_dir = "./model_Q_learning_3classes"
model_name = "model"
temp_model_dir = osp.join(model_dir, model_name+"-temp")
best_model_dir = osp.join(model_dir, model_name+"-best")
log_dir = "./log_Q_learning_3classes"
log_name = "log.txt"

# add a log message
def add_log(content):
    if not osp.isdir(log_dir):
        os.makedirs(log_dir)
        add_log("message:create folder '{}'".format(log_dir))
    log_file = osp.join(log_dir, log_name)
    print(content)
    tools.write_file(log_file, content, True)
    return

def do_actions(pre_actions, ls_batch_x, ls_batch_y):
    '''执行pre_actions做动作，删除对应的数据，0删除1保留'''
    pre_actions = np.argmax(pre_actions, axis=-1)
    ls_del_index = []
    ls_save_index = []  # 保留的部分
    for index in range(len(pre_actions)):
        if pre_actions[index] == 0:
            ls_del_index.append(index)
        else:
            ls_save_index.append(index)

    # do actions
    ls_batch_x_d, ls_batch_y_d = np.delete(ls_batch_x, ls_del_index, axis=0), np.delete(ls_batch_y, ls_del_index, axis=0)

    return ls_batch_x_d, ls_batch_y_d, ls_save_index

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
    data_test = Data_loader(test_data_file, num_classes, size, batch_size)

    global_step = tf.Variable(0, trainable=False, name="global_step")
    # global_step = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name="global_step")

    # srnet和tdnet的输入都是data
    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, size, 1], name="inputs")

    
    ''' ######### sr net ######### '''
    # 是reward，是一个数
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
    pred_outputs, td_pred_rewards = td_net.forward(inputs=inputs, actions=sr_pred_actions)
    print("td net 参数:{}k".format(td_net.get_parameter_num()/1000.0))

    ''' ######### loss td net ######### '''
    # td_net的更新使用交叉熵
    loss_td_net = Loss_TDNet_Q_learning().get_loss(y_pred=pred_outputs, y_true=td_net_y_true)
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

        # total_step = epoch_train * data.steps_per_epoch
        # while step < total_step:
        acc_max = 0
        while True:
            reward_total = 0
            td_loss_total = 0
            sr_loss_total = 0
            for _ in tqdm(range(data.steps_per_epoch)):
                x_batch, y_batch = next(data)
                # 1. 判断哪些标签应该被删掉，不能直接将两个网络放一起，会更新SRNet
                # 需要随机一定概率随机删除???
                if np.random.random() < 1:
                    [sr_actions] = sess.run([sr_pred_actions],feed_dict={inputs:x_batch})
                else:
                    sr_actions = np.expand_dims(np.random.randint(0, 2, batch_size), axis=-1)


                # 2. 删掉标签后，更新TDNet
                # _, reward, lr1 = sess.run([train_op_td_net, td_pred_rewards, lr_td_net],feed_dict={
                _, reward, lr1, td_loss = sess.run([train_op_td_net, td_pred_rewards, lr_td_net, loss_td_net],feed_dict={
                                                                                                        inputs:x_batch,
                                                                                                        td_net_y_true:y_batch,
                                                                                                        sr_pred_actions:sr_actions
                                                                                                    })
                td_loss_total += td_loss

                # 3. 更新SRNet
                _, lr2, step, sr_loss = sess.run([train_op_sr_net, lr_sr_net, global_step, loss_sr_net], feed_dict={
                                                                                                        inputs:x_batch,
                                                                                                        sr_y_true:reward
                                                                                                    })
                reward_total += reward
                sr_loss_total += sr_loss
                pass

            curr_epoch = np.math.floor(step/data.steps_per_epoch)
            add_log("message: epoch={}, lr1:{}, lr2:{}, avg_td_loss:{}, avg_sr_loss:{}, avg_reward:{}".format(
                                                                                curr_epoch, lr1, lr2, 
                                                                                td_loss_total/data.steps_per_epoch, 
                                                                                sr_loss_total/data.steps_per_epoch,
                                                                                reward_total/data.steps_per_epoch)
                                                                            )

            if curr_epoch % save_per_epoch == 0:
                # save ckpt model
                print("message: save ckpt model, step:{}, avg_reward:{}".format(step, reward_total/data.steps_per_epoch))
                saver.save(sess, osp.join(model_dir, model_name), global_step=step)

            # save temp_model
            print("message:saving the temp model, epoch:{}, step:{}, avg_reward:{}".format(curr_epoch, step, reward_total/data.steps_per_epoch))
            saver.save(sess, osp.join(temp_model_dir, model_name+"-temp"))

            # 测试
            count = 0
            for index in range(0, len(data_test), batch_size):
                x_batch, y_batch = data_test.get_data(range(index, min(index+batch_size, len(data_test))))
                [output] = sess.run([tf.argmax(pred_outputs, -1)],feed_dict={inputs:x_batch})
                y = np.argmax(y_batch, -1)
                count += np.sum(y==output)
                pass
            curr_acc = count*100/len(data_test)
            print("acc {}%".format(curr_acc))
            if curr_acc > acc_max:
                acc_max = curr_acc
                # save best temp_model
                add_log("message:saving the best model, epoch:{}, step:{}, acc:{}%".format(curr_epoch, step, curr_acc))
                saver.save(sess, osp.join(best_model_dir, model_name+"-{}".format(acc_max)))

    return

def main():
    if not osp.isdir(model_dir):
        os.makedirs(model_dir)
        print("建立文件夹 {}".format(model_dir))

    backward()
    return 

if __name__ == "__main__":
    main()