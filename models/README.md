# models 目录说明

本目录用于管理模型结构定义。

## 文件说明

- `mlp.py`：MLP 基线模型定义
- `cnn.py`：CNN 主模型定义
- `blocks.py`：可复用模块，如卷积块
- `initialize.py`：参数初始化方法

## 设计思路

- 保留 MLP 作为基线模型，便于和 CNN 对比
- CNN 作为主模型，面向 MNIST 图像分类任务
- 公共模块拆分到 `blocks.py`，便于复用和扩展

