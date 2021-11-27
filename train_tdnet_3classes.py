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
num_classes = 3 
train_data_file = config.train_data_file
test_data_file = "./Data/testdata-taskB-all-annotations.txt"
lr_base = config.lr_base
lr_min = config.lr_min
save_per_epoch = config.save_per_epoch
warm_epoch = config.warm_epoch
train_epoch = config.train_epoch
model_dir = "./model_tdnet_3classes"
model_name = "model"
temp_model_dir = osp.join(model_dir, model_name+"-temp")
best_model_dir = osp.join(model_dir, model_name+"-best")
log_dir = "./log_tdnet_3classes"
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

def backward():
    data = Data_loader(train_data_file, num_classes, size, batch_size)
    data_test = Data_loader(test_data_file, num_classes, size, batch_size)
    iterator, inputs, td_net_y_true = data.init_tf_dataset()

    global_step = tf.Variable(0, trainable=False, name="global_step")

    td_net = TDNet_Q_learning(num_classes=num_classes)
    pred_outputs, _ = td_net.forward(inputs=inputs, actions=np.zeros((batch_size, 1), dtype=np.float32))
    print("td net 参数:{}k".format(td_net.get_parameter_num()/1000.0))

    # td_net的更新使用交叉熵
    loss_td_net = Loss_TDNet_Q_learning().get_loss(y_pred=pred_outputs, y_true=td_net_y_true)
    lr_td_net = Learning_rate(lr_base, lr_min=lr_min).warmup_Cosine_lr(
                                                                        global_step=global_step,
                                                                        total_step=data.steps_per_epoch * train_epoch,
                                                                        warmup_step=warm_epoch*data.steps_per_epoch
                                                                    )
    optimizer_td_net = Optimizer(lr_td_net).adam()
    update_ops_td_net = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_td_net):
        gvs = optimizer_td_net.compute_gradients(loss_td_net)
        clip_grad_var = [gv if gv[0] is None else[tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op_td_net = optimizer_td_net.apply_gradients(clip_grad_var, global_step=global_step)

    saver = tf.compat.v1.train.Saver(max_to_keep=250)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(iterator.initializer)

        ckpt = tf.compat.v1.train.get_checkpoint_state(temp_model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print("message:can not fint ckpt model")

        step = 0
        acc_max = 0
        total_step = train_epoch * data.steps_per_epoch
        while step < total_step:
            td_loss_total = 0
            for _ in tqdm(range(data.steps_per_epoch)):

                _, td_loss, lr1, step = sess.run([train_op_td_net, loss_td_net, lr_td_net, global_step])
                td_loss_total += td_loss
                pass

            curr_epoch = np.math.floor(step/data.steps_per_epoch)
            add_log("message: epoch={}, lr:{}, avg_td_loss:{}".format(
                                                                                curr_epoch, lr1, 
                                                                                td_loss_total/data.steps_per_epoch
                                                                                )
                                                                            )

            if curr_epoch % save_per_epoch == 0:
                # save ckpt model
                print("message: save ckpt model, step:{}".format(step))
                saver.save(sess, osp.join(model_dir, model_name), global_step=step)

            # save temp_model
            print("message:saving the temp model, epoch:{}, step:{}".format(curr_epoch, step))
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