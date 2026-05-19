# 文件与目录说明

## 根目录文件

- `README.md`：项目入口说明文档
- `.gitignore`：Git 忽略规则
- `requirements.txt`：项目依赖列表
- `main.py`：训练入口脚本
- `evaluate.py`：测试评估入口
- `infer.py`：推理与预测结果展示入口

---

## docs/

- `PROJECT_OVERVIEW.md`：项目整体设计说明
- `FILES_REFERENCE.md`：文件与目录职责说明
- `CONFIG_GUIDE.md`：配置项说明
- `EXPERIMENT_GUIDE.md`：实验设计说明
- `GITHUB_GUIDE.md`：GitHub 使用与工程规范说明
- `assets/report/`：随仓库提交的正式报告图片

---

## configs/

- `README.md`：配置目录说明
- `base.yaml`：公共基础配置
- `mlp_baseline.yaml`：MLP 基线实验配置
- `cnn_basic.yaml`：基础 CNN 实验配置
- `cnn_basic_long.yaml`：基础 CNN 长训练配置
- `cnn_basic_long_report.yaml`：用于 CPU / 本地报告复现的报告专用配置
- `cnn_bn_adamw.yaml`：加入 BatchNorm 与 AdamW 的配置
- `cnn_aug.yaml`：加入数据增强的配置

---

## data/

- `README.md`：数据模块说明
- `transforms.py`：定义数据预处理与增强策略
- `loader.py`：构建训练集、验证集、测试集 DataLoader

---

## models/

- `README.md`：模型模块说明
- `mlp.py`：MLP 模型定义
- `cnn.py`：CNN 主模型定义
- `cnn_residual.py`：带残差连接的 CNN 模型
- `blocks.py`：可复用基础模块
- `factory.py`：根据配置统一构建模型
- `initialize.py`：参数初始化方法

---

## engine/

- `README.md`：训练引擎说明
- `trainer.py`：训练流程实现
- `evaluator.py`：评估流程实现
- `losses.py`：损失函数封装

---

## optimizers/

- `README.md`：优化器模块说明
- `builder.py`：根据配置构建优化器
- `scheduler.py`：学习率调度策略

---

## utils/

- `README.md`：工具模块说明
- `seed.py`：随机种子管理
- `logger.py`：日志管理
- `metrics.py`：指标计算
- `checkpoint.py`：模型保存与加载
- `visualize.py`：训练曲线和预测图像可视化

---

## outputs/

- `checkpoints/`：模型权重文件，仅本地保存，不纳入 Git
- `figures/`：损失曲线、准确率曲线等图像，仅本地保存，不纳入 Git
- `logs/`：训练日志，仅本地保存，不纳入 Git
- `predictions/`：推理结果图，仅本地保存，不纳入 Git

---

## report/

- `README.md`：报告目录使用说明
- `report.md`：正式实验报告
- `report_for_docx_v2.md`：当前保留的 Word 导出源文件
- `experiment_log.md`：实验记录表

说明：`report/` 中推荐只保留 Markdown 源文件，不建议将生成的 `.docx`、`report/assets/` 图像副本或其他中间产物提交到 GitHub。
