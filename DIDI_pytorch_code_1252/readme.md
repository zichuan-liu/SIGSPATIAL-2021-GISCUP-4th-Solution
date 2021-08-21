## 运行文件

1. 打开配置文件 `config.yml`, 设置参数
| 参数 | 地址 |
| ---- | ---- |
| data_dir | 原始数据地址 |
| pkl_dir | 生成pkl文件存放地址 |
| msg_dir | 生成msg文件存放地址 |


2. 运行 `create_train_data.ipynb`生成训练数据

3. 运行`model.ipynb`训练模型

4. 打开 `predict.ipynb`，设置`ckpt_path` 后, 运行 文件, 将在`predict`目录生成预测的csv文件


## 模型简介

将link序列相关特征拼接后通过LSTM，输出向量再和其他特征拼接，最后经过MLP得到预测结果

#### link序列相关特征
| 特征 | dim |描述|
| ---- | ---- | ---- |
| link_cross_id | 20 |link_id和cross_id拼接到一起link_cross_id|
| link_cross_time | 20 |link_time和cross_time拼接到一起组成link_cross_time|
| current_status | 1 ||
| link_ratio | 1 ||

#### 其他特征
| 特征 | dim |
| ---- | ---- |
| driver | 20 |
| weekday | 3|
| simple_eta | 1 |
| dist | 1 |
| low_temp | 1 |
| high_temp | 1 |

