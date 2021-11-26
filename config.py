# coding:utf-8
# 配置文件
size = 24
batch_size = 32
num_classes = 3
epoch_total = 200
warm_epoch = 10
lr_base = 0.00125 * batch_size/64
lr_min = lr_base/50.0

train_data_file = "Data/trainingdata-all-annotations.txt"