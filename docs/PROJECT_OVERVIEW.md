# Project Overview

## 1. 项目背景

本项目是一个基于 PyTorch 的 MNIST 手写数字识别课程作业项目。  
项目以上一版完成的 **MLP baseline** 为起点，重新构建并比较 **CNN、Adam/AdamW、Dropout、Weight Decay、Initialization、BatchNorm、Data Augmentation、Skip Connection** 等常见训练与结构改进方法在 MNIST 分类任务中的效果。

项目不仅关注模型训练结果，还强调：

- 配置驱动的实验管理
- 模块化工程结构
- 训练、评估、推理三条主链路完整
- 结果可记录、可复现、可分析

---

## 2. 项目目标

本项目的主要目标包括：

1. 以 MLP 模型作为基线，构建可比较的起点；
2. 实现基于 CNN 的手写数字识别模型；
3. 尽可能在统一项目框架中纳入课程中提到的方法；
4. 对不同方法进行实验对照与结果分析；
5. 形成一个适合 GitHub 管理和课程展示的完整项目。

---

## 3. 项目主线

本项目的实验主线遵循“**先完成主结构改进，再逐步扩展训练技巧**”的思路：

- 先复现 MLP baseline
- 再将模型结构替换为 CNN
- 在 CNN 基础上尝试：
  - Adam / AdamW
  - Dropout
  - Weight Decay
  - BatchNorm
  - Data Augmentation
  - Skip Connection
- 最后对表现较好的模型做更充分训练

这种设计使得实验对比逻辑更清晰，也更便于分析“哪些方法真正有效”。

---

## 4. 当前主要结论

根据当前实验结果，可以得到以下结论：

1. **CNN 相比 MLP 在 MNIST 任务上具有明显优势**；
2. **从 MLP 转换为 CNN 是性能提升最显著的一步**；
3. AdamW、BatchNorm、Dropout、Weight Decay、Data Augmentation、Skip Connection 等方法具有实验价值，但在当前设置下并未超过基础 CNN 长训练方案；
4. 当前最优模型为 `cnn_basic_long`，其最佳验证准确率达到 `0.9924`，测试集准确率达到 `0.9928`；
5. 项目已完成训练、评估和推理三条主链路。

---

## 5. 工程结构特点

本项目采用模块化组织方式，主要特点如下：

- `configs/`：统一管理实验配置
- `data/`：负责数据预处理与数据加载
- `models/`：负责模型定义
- `engine/`：负责训练与评估逻辑
- `optimizers/`：负责优化器与调度器构建
- `utils/`：负责日志、随机种子、可视化、配置读取等公共工具
- `outputs/`：存放 checkpoint、日志、图像和推理结果
- `report/`：存放实验记录与正式报告

这种结构有利于：

- 清晰组织实验
- 降低代码耦合
- 提高可读性与可维护性
- 为后续扩展预留空间

---

## 6. 可扩展方向

虽然当前项目已经完成了主要实验主线，但仍保留了若干可扩展方向，例如：

- More training data
- Semi-supervised
- 更复杂的 Parameter Regularization
- 更完整的 learning rate scheduler
- 更深层的残差结构

因此，本项目不仅完成了课程作业要求的核心部分，也为后续进一步优化预留了接口和空间。

