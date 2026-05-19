# 实验设计说明

## 1. 实验目的

本项目希望通过对比不同模型结构与训练策略，分析它们对 MNIST 手写数字识别任务的影响。

---

## 2. 实验主线

### 实验 1：MLP Baseline
目的：复现前期实验结果，作为后续 CNN 对比基线。

### 实验 2：CNN Basic
目的：验证仅改变模型结构为 CNN 后是否能够提升性能。

### 实验 3：CNN + Dropout + Weight Decay
目的：验证正则化方法对泛化能力的帮助。

### 实验 4：CNN + BatchNorm + AdamW
目的：观察归一化与优化器改进对训练稳定性和收敛效果的影响。

### 实验 5：CNN + Data Augmentation
目的：进一步提升模型泛化能力。

---

## 3. 建议记录内容

每组实验建议至少记录以下内容：

- 使用的配置文件
- 模型名称
- 优化器名称
- 学习率
- Batch Size
- 是否使用 Dropout
- 是否使用 Weight Decay
- 是否使用 BatchNorm
- 是否使用数据增强
- 训练轮数
- 最佳准确率
- 实验现象与结论

---

## 4. 实验对比建议

建议最终至少给出以下对比：

- MLP vs CNN
- CNN Basic vs CNN + Regularization
- CNN Basic vs CNN + BatchNorm
- CNN Basic vs CNN + AdamW
- CNN Basic vs CNN + Data Augmentation

