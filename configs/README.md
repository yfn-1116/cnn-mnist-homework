# configs 目录说明

本目录用于管理所有实验配置文件。  
训练、评估和推理脚本都会通过 `--config` 读取这里的配置。

## 主要配置文件

- `base.yaml`：公共基础配置
- `mlp_baseline.yaml`：MLP 基线实验
- `cnn_basic.yaml`：基础 CNN 实验
- `cnn_basic_long.yaml`：基础 CNN 的长训练版本
- `cnn_basic_long_report.yaml`：为报告复现准备的 CPU / 单线程版本
- `cnn_reg.yaml`：加入正则化相关设置的 CNN 实验
- `cnn_bn_adamw.yaml`：加入 BatchNorm 和 AdamW 的实验
- `cnn_aug.yaml`：加入数据增强的实验
- `cnn_aug_tuned.yaml`：调优后的数据增强实验
- `cnn_skip.yaml`：带跳连结构的 CNN 实验
- `cnn_skip_long.yaml`：跳连结构的长训练版本

## 使用方式

训练：

```bash
python main.py --config configs/cnn_basic.yaml
```

评估：

```bash
python evaluate.py --config configs/cnn_basic.yaml --checkpoint outputs/checkpoints/cnn_basic_best.pth --split test
```

推理：

```bash
python infer.py --config configs/cnn_basic.yaml --checkpoint outputs/checkpoints/cnn_basic_best.pth --image path/to/image.png
```

## 配置内容

配置文件通常包含以下几类参数：

- `data`：数据路径、batch size、是否归一化、是否增强
- `model`：模型名称、层参数、dropout、batchnorm、初始化方式
- `train`：训练轮数、学习率等训练超参数
- `optimizer`：优化器类型
- `scheduler`：是否启用学习率调度器
- `output`：checkpoint、日志、图像和推理结果的输出目录

更详细的字段说明可参考 `docs/CONFIG_GUIDE.md`。
