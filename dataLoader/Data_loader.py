# coding:utf-8
# 数据加载器
from utils import tools
import logging
import tqdm
import bitermplus as btm
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE

class Data_loader():
    def __init__(self, file_name, num_classes, size=25, batch_size=64) -> None:
        self.file_name = file_name
        self.num_classes = num_classes
        self.size = size
        self.batch_size = batch_size

        self.map_stance = {
            "AGAINST":[0.99, 0.005, 0.005],
            "NONE":[0.005, 0.99, 0.005],
            "FAVOR":[0.005, 0.005, 0.99]
        }

        self.__init_args()
        self.ls_padding_vector_tweet = self.padding_vector_word()
        self.ls_labels = self.make_labels()
        # self.tongji()
        pass

    def __init_args(self):
        '''
        初始化参数
        '''
        ls_file_content = tools.read_file(self.file_name)
        assert ls_file_content is not None, "file {} is illegal".format(self.file_name)
        
        self.ls_id = []
        self.ls_target = []
        self.ls_tweet = []
        self.ls_stance = []
        self.ls_opinion_towards = []
        self.ls_sentiment = []

        # split
        for line in tqdm.tqdm(ls_file_content[1:]):
            ls_line = line.strip().split('\t')
            # ID	Target	Tweet	Stance	Opinion towards	Sentiment
            assert len(ls_line)==6, "the result of split is illegal, expected {} but get {}".format(6, len(ls_line))
            self.ls_id.append(ls_line[0])
            self.ls_target.append(ls_line[1])
            self.ls_tweet.append(ls_line[2])
            self.ls_stance.append(ls_line[3])
            self.ls_opinion_towards.append(ls_line[4])
            self.ls_sentiment.append(ls_line[5])
            pass

        # 向量化
        X, self.map_id2word, self.map_word2id = btm.get_words_freqs(self.ls_tweet)
        self.ls_vectorized_tweet = btm.get_vectorized_docs(self.ls_tweet, self.map_word2id)   # Compute words vs documents frequency matrix.
        
        # 一个epoch多少step
        self.steps_per_epoch = len(self.ls_vectorized_tweet)//self.batch_size

        return None

    def tongji(self):
        '''统计词条'''
        # 统计每一句的长度
        ls_length = [len(x) for x in self.ls_vectorized_tweet]
        ls_x = [x for x in range(len(self.ls_vectorized_tweet))]
        logging.info("最长 {} 最短 {}".format(max(ls_length), min(ls_length)))
        plt.scatter(ls_x, ls_length)
        plt.show()
        return None

    def padding_vector_word(self, padding_value=0):
        '''把单词向量padding 0'''
        ls_result = []
        for line in tqdm.tqdm(self.ls_vectorized_tweet):
            curr = [padding_value] * self.size
            curr[0: min(self.size, len(line))] = line
            ls_result.append(curr)
        return ls_result

    def make_labels(self):
        '''制作stance对应的标签'''
        # self.ls_stance
        ls_result = [self.map_stance[stance] for stance in self.ls_stance]
        return ls_result

    def __len__(self):
        return len(self.ls_tweet)

    def __next__(self):
        '''next'''
        return self.get_data()

    def get_data(self, ls_index=None):
        '''得到给的index的数据'''
        if ls_index is None:
            ls_index = np.random.randint(0, len(self), size=self.batch_size)

        ls_batch_x = []
        ls_batch_y = []
        for index in ls_index:
            x = self.ls_padding_vector_tweet[index]
            y = self.ls_labels[index]
            ls_batch_x.append(x)
            ls_batch_y.append(y)

        ls_batch_x = np.asarray(ls_batch_x).astype(np.float32)
        ls_batch_x = np.expand_dims(ls_batch_x, -1)
        ls_batch_y = np.asarray(ls_batch_y).astype(np.float32)
        return ls_batch_x, ls_batch_y

    def load_tf_batch_data(self, tf_inputs):
        '''
            inp=[(imgs_batch, xmls_batch)],
            Tout=[tf.float32, tf.float32, tf.float32]),
        '''
        # use tf.data to load the image and label
        ls_index = tf_inputs.tolist()
        # imgs_batch, xmls_batch = tf_inputs
        # decode_type = 'UTF-8'            
        # # label_name = xmls_batch[i]
        # label_name = label_name.decode(decode_type)
        # img_name = img_name.decode(decode_type)     # chinese also working

        return self.get_data(ls_index=ls_index)

    def init_tf_dataset(self):
        '''create dataset'''
        ls_folder_id = [x for x in range(len(self))]
        dataset = tf.data.Dataset.from_tensor_slices(ls_folder_id)
        dataset = dataset.shuffle(len(ls_folder_id))   # shuffle
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=self.batch_size)    #
        dataset = dataset.map(
            lambda ls_inputs_id: tf.py_func(
                self.load_tf_batch_data,
                inp=[ls_inputs_id],
                Tout=[tf.float32, tf.float32]),
            num_parallel_calls=4
        )
        dataset = dataset.prefetch(AUTOTUNE)

        # iterator
        iterator = dataset.make_initializable_iterator()

        # set shape
        inputs, labels = iterator.get_next()
        inputs.set_shape([None, self.size, 1])
        labels.set_shape([None, self.num_classes])

        return iterator, inputs, labels