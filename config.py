# coding:utf-8
# 配置文件
size = 24
batch_size = 32
num_classes = 3
train_epoch = 500
save_per_epoch = 50
warm_epoch = 10
lr_base = 0.00125 * batch_size/64
lr_min = lr_base/50.0

train_data_file = "Data/trainingdata-all-annotations.txt"