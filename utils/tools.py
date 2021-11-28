# coding:utf-8
# 工具函数
import os.path as osp
import time
import numpy as np

'''
##################### 文件操作 #####################
'''
# 读取文件全部内容
def read_file(file_name):
    '''
    读取 file_name 文件全部内容
    return:文件内容list
    '''
    if not osp.isfile(file_name):
        return None
    result = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            # 去掉换行符和空格
            line = line.strip('\n').strip()
            if len(line) == 0:
                continue
            result.append(line)
    return result

# 写入文件,是否写入时间
def write_file(file_name, line, write_time=False):
    '''
    file_name:写入文件名
    line:写入文件内容
    write_time:是否在内容前一行写入时间
    '''
    with open(file_name,'a') as f:
        if write_time:
            line = get_curr_date() + '\n' + str(line)
        f.write(str(line) + '\n')
    return None

'''
######################## 时间操作 ####################
'''
# 获得当前日期
def get_curr_date():
    '''
    return : 年-月-日-时-分-秒
    '''
    t = time.gmtime()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S",t)
    return time_str

'''
######################## 准确率等 ####################
'''
def f1(y_pred, y_true, num_classes):
    '''
    y_pred:[batch, num_classes]
    y_true:[batch, num_classes]
    num_classes:分类数
    精确率:我对了多少
    召回率:我预测为对的对了多少
    '''
    arr_pred = np.argmax(y_pred, -1)
    arr_label = np.argmax(y_true, -1)
    # arr_pred = y_pred
    # arr_label = y_true
    ls_result = []

    for index in range(num_classes):
        # 计算每一类的f1分数
        # 标签里面有多少个 index ,和其index
        curr_label_index = np.where(arr_label==index)
        curr_label = arr_label[curr_label_index]
        curr_pred = arr_pred[curr_label_index]
        # 预测对了几个index
        num_right = np.sum(curr_label==curr_pred)
        # 预测了几个index
        curr_pred_index = len(np.where(arr_pred==index)[0])
        # 精度
        curr_precision = 0 if len(curr_label)==0 else (num_right*1.0 / len(curr_label))
        # 召回率
        curr_recall = 0 if curr_pred_index==0 else (num_right*1.0 / curr_pred_index)
        ls_result.append([curr_precision, curr_recall])
        pass

    # 计算f1
    ls_f1 = [0 if (x[0]+x[1])==0 else ((2*x[0]*x[1])/(x[0]+x[1])) for x in ls_result]

    return np.mean(ls_f1), ls_result