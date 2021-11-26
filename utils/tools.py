# coding:utf-8
# 工具函数
import os.path as osp
import time

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
