# 实验记录表

| 实验编号 | 日期 | 配置文件 | 模型 | 优化器 | LR | Batch Size | Dropout | Weight Decay | BatchNorm | Data Aug | Epochs | Best Acc | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---|
| Exp-01 | | mlp_baseline.yaml | MLP | Adam | | | | | No | No | | | 基线实验 |
| Exp-02 | | cnn_basic.yaml | CNN | Adam | | | | | No | No | | | 基础 CNN |
| Exp-03 | | cnn_bn_adamw.yaml | CNN | AdamW | | | | | Yes | No | | | 加入 BN 与 AdamW |
| Exp-04 | | cnn_aug.yaml | CNN | AdamW | | | | | Yes | Yes | | | 加入数据增强 |


## EXP-001
- date: 2026-04-15
- config: configs/mlp_baseline.yaml
- model: MLP
- epochs: 10
- best_val_acc: 0.9786
- note: baseline 跑通，超过上次约 97%

## EXP-002
- date: 2026-04-15
- config: configs/cnn_basic.yaml
- model: CNN
- epochs: 10
- best_val_acc: 0.9918
- note: cnn basic 跑通，显著高于 mlp_baseline 的 0.9786

## EXP-003
- date: 2026-04-15
- config: configs/cnn_bn_adamw.yaml
- model: CNN
- optimizer: AdamW
- epochs: 10
- best_val_acc: 0.9890
- note: 将原 15 轮统一改为 10 轮后重新训练；公平对比下低于 cnn_basic 的 0.9918

## EXP-004
- date: 2026-04-15
- config: configs/cnn_aug.yaml
- model: CNN
- epochs: 10
- best_val_acc: 0.9890
- note: 本次结果与 cnn_bn_adamw 基本一致，怀疑 augmentation 配置或数据增强逻辑未实际生效，需先检查配置与代码

## EXP-004-DEBUG
- date: 2026-04-15
- issue: cnn_aug 与 cnn_bn_adamw 结果逐轮完全一致
- cause: random_split 后 train/val 共享同一底层 dataset；设置 val_dataset.dataset.transform = test_transform 时覆盖了 train transform，导致 augmentation 实际未生效
- action: 重构 data/loader.py，分别为 train 和 val 创建独立数据集对象，并复用相同索引划分

## EXP-004
- date: 2026-04-15
- config: configs/cnn_aug.yaml
- model: CNN
- optimizer: AdamW
- epochs: 10
- best_val_acc: 0.9910
- note: 修复 data augmentation 未生效的问题后重新训练；结果高于 cnn_bn_adamw 的 0.9890，但仍略低于 cnn_basic 的 0.9918

## EXP-004-DEBUG
- date: 2026-04-15
- issue: cnn_aug 与 cnn_bn_adamw 结果逐轮一致
- cause: random_split 后 train/val 共享同一底层 dataset；设置 val_dataset.dataset.transform = test_transform 时覆盖了 train transform，导致 augmentation 未实际生效
- fix: 为 train 和 val 分别创建独立 MNIST 数据集对象，并复用相同索引划分
- rerun_result: cnn_aug best_val_acc = 0.9910

## EXP-005
- date: 2026-04-15
- config: configs/cnn_reg.yaml
- model: CNN
- optimizer: Adam
- epochs: 10
- best_val_acc: 0.9902
- note: 仅加入 Dropout + Weight Decay 的正则化对照实验；结果高于 cnn_bn_adamw 的 0.9890，但低于 cnn_basic 的 0.9918 和 cnn_aug 的 0.9910

## EXP-006
- date: 2026-04-15
- config: configs/cnn_aug_tuned.yaml
- model: CNN
- optimizer: AdamW
- epochs: 10
- best_val_acc: 0.9890
- note: 调整为更小学习率、较小 weight decay 和 dropout=0.3 后重新实验；结果未超过 cnn_basic，说明当前调参方向未带来提升

## EXP-007
- date: 2026-04-15
- config: configs/cnn_basic_long.yaml
- model: CNN
- optimizer: Adam
- epochs: 15
- best_val_acc: 0.9924
- note: 在 cnn_basic 基础上延长训练轮数到 15；结果超过原 cnn_basic 的 0.9918，成为当前最佳结果

## EXP-008
- date: 2026-04-16
- config: configs/cnn_skip.yaml
- model: CNN + Skip Connection
- optimizer: Adam
- epochs: 10
- best_val_acc: 0.9910
- note: 引入轻量残差块（Skip Connection）；结果未超过 cnn_basic 的 0.9918，说明在当前浅层网络下跳连优势不明显

## EXP-009
- date: 2026-04-16
- config: configs/cnn_skip_long.yaml
- model: CNN + Skip Connection
- optimizer: Adam
- epochs: 15
- best_val_acc: 0.9916
- note: 对 Skip Connection 模型延长训练到 15 轮后，结果仍未超过 cnn_basic_long 的 0.9924，说明跳连在当前浅层 MNIST 模型中的增益有限

## EXP-009
- date: 2026-04-16
- config: configs/cnn_skip_long.yaml
- model: CNN + Skip Connection
- optimizer: Adam
- epochs: 15
- best_val_acc: 0.9916
- note: 对 Skip Connection 模型延长训练到 15 轮后，结果仍未超过 cnn_basic_long 的 0.9924，说明跳连在当前浅层 MNIST 模型中的增益有限

## EXP-009
- date: 2026-04-16
- config: configs/cnn_skip_long.yaml
- model: CNN + Skip Connection
- optimizer: Adam
- epochs: 15
- best_val_acc: 0.9916
- note: 对 Skip Connection 模型延长训练到 15 轮后，结果仍未超过 cnn_basic_long 的 0.9924，说明跳连在当前浅层 MNIST 模型中的增益有限

## EXP-010
- date: 2026-04-16
- config: configs/cnn_basic_long.yaml
- checkpoint: outputs/checkpoints/cnn_basic_long_best.pth
- split: test
- test_loss: 0.0344
- test_acc: 0.9928
- note: 使用当前最佳模型在测试集上评估，测试准确率达到 0.9928

## EXP-011
- date: 2026-04-16
- config: configs/cnn_basic_long.yaml
- checkpoint: outputs/checkpoints/cnn_basic_long_best.pth
- task: inference
- image: outputs/predictions/mnist_test0_label_7.png
- note: 使用当前最佳模型完成单张图片推理，结果见 outputs/predictions/ 目录
