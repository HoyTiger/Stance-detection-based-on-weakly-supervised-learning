# coding:utf-8
# 学习率
import tensorflow as tf
import numpy as np

class Learning_rate():
    def __init__(self, lr_base, lr_min=1e-5) -> None:
        self.lr_base = lr_base
        self.lr_min = lr_min
        pass

    def piecewise_constance(self, epoch, piecewise_boundaries, piecewise_values):
        lr = tf.compat.v1.train.piecewise_constant(
                                                    epoch,
                                                    piecewise_boundaries,
                                                    piecewise_values
                                                )
        return tf.maximum(lr, self.lr_min)

    def warmup_Cosine_lr(self, global_step, total_step, warmup_step=0, keep_step=0):
        '''
        lr_base:基本学习率
        global_step:当前步数
        total_step:总步数
        warmup_step:持续增加阶段的步数
        keep_step:保持阶段的步数
        lr_min:最小学习率
        '''
        lr_base = self.lr_base
        lr_min=self.lr_min
        learn_rate = tf.cond(
            pred=tf.less(global_step, warmup_step),
            true_fn=lambda:lr_min + global_step / warmup_step * (lr_base - lr_min),
            false_fn=lambda:tf.cond(
                pred=global_step < warmup_step + keep_step,
                true_fn=lambda:tf.constant(lr_base, dtype=tf.float64),
                false_fn=lambda:tf.cond(
                    pred=global_step >= total_step,
                    true_fn=lambda:tf.cast(lr_min, tf.float64),
                    false_fn=lambda:lr_min + 0.5 * (lr_base-lr_min) * (
                        1.0 + tf.cos((global_step-warmup_step-keep_step) / (total_step-warmup_step-keep_step) * np.pi
                        )
                    )
                )
            )
        )
        return learn_rate
