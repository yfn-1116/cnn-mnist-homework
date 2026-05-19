# 配置文件说明

本项目所有实验参数均通过 `configs/*.yaml` 管理。

---

## 1. data

- `dataset`：数据集名称
- `root`：数据集保存路径
- `batch_size`：每个 batch 的样本数量
- `num_workers`：DataLoader 线程数
- `normalize`：是否进行归一化
- `augmentation`：是否启用数据增强

---

## 2. model

- `name`：模型名称，支持 `mlp` 或 `cnn`
- `input_dim`：MLP 输入维度
- `hidden_dims`：MLP 隐藏层维度
- `num_classes`：分类类别数
- `dropout`：Dropout 比例
- `use_batchnorm`：是否启用 BatchNorm
- `init_method`：参数初始化方法，如 `he`

---

## 3. train

- `epochs`：训练轮数
- `lr`：学习率
- `weight_decay`：权重衰减系数

---

## 4. optimizer

- `name`：优化器名称，如 `sgd`、`adam`、`adamw`

---

## 5. scheduler

- `use`：是否启用学习率调度器

---

## 6. output

- `checkpoint_dir`：模型权重保存目录
- `figure_dir`：图像输出目录
- `log_dir`：日志输出目录
- `prediction_dir`：推理结果输出目录

