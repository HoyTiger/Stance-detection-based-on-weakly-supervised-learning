# Stance-detection-based-on-weakly-supervised-learning
## 强化学习
* training
``` bash
python train_Q_learning_2classes.py
```
* inference
``` bash
python inference_Q_learning_2classes.py
```
* 加强化学习后的F1分数是58.758
## 不加强化学习
* training
``` bash
python train_tdnet_2classes.py
```
* inference
``` bash
python inference_tdnet_2classes.py
```
* 不加强化学习后的F1分数是53.762

* **以上结果还有提升空间**