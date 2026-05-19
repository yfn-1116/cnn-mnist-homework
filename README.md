# CNN MNIST Homework

基于 PyTorch 的 MNIST 手写数字识别课程项目。这个仓库不是只放一个 `CNN` 模型，而是围绕“从 `MLP baseline` 到 `CNN` 及其训练技巧对比”组织的一套完整实验工程，包含训练、评估、推理、实验配置、结果图表和课程报告。

## 1. 这个项目在做什么

项目主线很直接：

1. 先用 `MLP` 做基线；
2. 再切换到 `CNN`；
3. 然后逐步加入 `BatchNorm`、`AdamW`、`Dropout`、`Weight Decay`、`Data Augmentation`、`Skip Connection` 等常见技巧；
4. 最后对效果较好的配置做更长时间训练，并整理报告。

当前报告中的最佳实验是 `cnn_basic_long`。

## 2. 项目结构

```text
cnn-mnist-homework/
├── configs/        # 实验配置：每个 yaml 对应一个实验
├── data/           # 数据加载与数据增强
├── docs/           # 补充说明文档
├── engine/         # 训练与评估核心流程
├── models/         # MLP / CNN / Residual CNN 定义
├── optimizers/     # optimizer 与 scheduler 构建
├── outputs/        # 本地产物：checkpoint、日志、图像、推理结果
├── report/         # 实验报告与实验记录
├── scripts/        # 生成图表和辅助脚本
├── utils/          # 配置、日志、随机种子、可视化、checkpoint 等工具
├── main.py         # 训练入口
├── evaluate.py     # 评估入口
├── infer.py        # 推理入口
└── requirements.txt
```

如果只想快速读懂项目，优先看这几处：

- `main.py`：训练主入口
- `models/factory.py`：按配置统一构建模型
- `data/loader.py`：构建 train / val / test dataloader
- `engine/trainer.py`：训练循环与最佳模型保存
- `report/report.md`：课程报告主文

## 3. 代码是怎么串起来的

### 训练链路

`main.py`
-> `utils.config.load_config`
-> `data.loader.build_dataloaders`
-> `models.factory.build_model`
-> `engine.losses.build_loss`
-> `optimizers.builder.build_optimizer`
-> `optimizers.scheduler.build_scheduler`
-> `engine.trainer.train`
-> `utils.visualize.plot_training_curves`

### 评估链路

`evaluate.py`
-> 加载配置和 checkpoint
-> 复用 dataloader 与模型构建逻辑
-> `engine.evaluator.evaluate`

### 推理链路

`infer.py`
-> 加载配置和 checkpoint
-> 读取单张图片并做预处理
-> 前向推理
-> 将结果保存到 `outputs/predictions/`

## 4. 各目录职责

### `configs/`

这里是实验管理中心。每个配置文件描述一次实验，比如：

- `mlp_baseline.yaml`：MLP 基线
- `cnn_basic.yaml`：基础 CNN
- `cnn_bn_adamw.yaml`：CNN + BatchNorm + AdamW
- `cnn_aug.yaml`：CNN + 数据增强
- `cnn_reg.yaml`：CNN + Dropout + Weight Decay
- `cnn_skip.yaml`：CNN + Skip Connection
- `cnn_basic_long.yaml`：基础 CNN 延长训练

### `models/`

模型定义被拆成了几层：

- `mlp.py`：全连接基线模型
- `cnn.py`：基础卷积模型
- `cnn_residual.py`：带残差块的 CNN
- `blocks.py`：基础卷积模块
- `factory.py`：根据配置统一构建模型

### `engine/`

这里放训练和评估过程：

- `trainer.py`：单轮训练、训练总循环、最优 checkpoint 保存
- `evaluator.py`：loss / accuracy 评估
- `losses.py`：损失函数构建

### `utils/`

这里放公共基础设施：

- `config.py`：读取 yaml 配置
- `logger.py`：实验日志
- `seed.py`：随机种子固定
- `checkpoint.py`：模型保存
- `visualize.py`：训练曲线和图像输出

## 5. 怎么运行

安装依赖：

```bash
pip install -r requirements.txt
```

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

## 6. 这个项目的实验逻辑

整体不是无序堆实验，而是按“模型结构优先，训练技巧次之”的顺序推进：

1. `mlp_baseline` 建立起点；
2. `cnn_basic` 验证卷积结构的收益；
3. 在 CNN 上逐步加训练技巧和正则化；
4. 对表现较好的方案做长训练；
5. 把结果沉淀到 `report/` 和 `outputs/figures/`。

这也是为什么从结果上看，`MLP -> CNN` 往往是最大的一次性能跃迁，而后续优化方法更多是在做细节增益和稳定性比较。

## 7. 当前整理后的结论

从工程角度看，这个项目已经具备比较完整的课程作业形态：

- 有清晰的模块分层；
- 有配置驱动实验；
- 有训练、评估、推理三条完整链路；
- 有报告、实验记录和图表产物；
- 有进一步扩展到更深网络或更多实验的空间。

从代码角度看，这次整理后最重要的收口是：模型构建逻辑已经统一到 `models/factory.py`，避免了 `main.py`、`evaluate.py`、`infer.py` 三份脚本各自维护一套相同逻辑。

报告文件目前也做了收口：

- `report/report.md`：仓库中的正式报告主文
- `report/report_for_docx_v2.md`：Word 导出源
- `report/experiment_log.md`：实验记录
- `docs/assets/report/`：随仓库提交的少量正式报告图片
- `report/assets/`：仅本地生成，不提交到 GitHub

## 8. 建议你接下来怎么用这个仓库

如果你是为了交作业或答辩，推荐按这个顺序看：

1. 先读 `README.md` 和 `docs/PROJECT_OVERVIEW.md`
2. 再看 `report/report.md`
3. 然后看 `configs/` 里的实验配置
4. 最后再回到 `main.py`、`models/`、`engine/` 理解代码实现

如果你是为了继续扩展：

1. 先从新增一个配置文件开始；
2. 如果需要新结构，再在 `models/` 新增模块并接入 `models/factory.py`；
3. 训练产物统一写入 `outputs/`；
4. 报告结论更新到 `report/`。

## 9. 相关文档

- [项目概览](docs/PROJECT_OVERVIEW.md)
- [文件说明](docs/FILES_REFERENCE.md)
- [配置说明](docs/CONFIG_GUIDE.md)
- [实验说明](docs/EXPERIMENT_GUIDE.md)
- [Git/GitHub 维护说明](docs/GITHUB_GUIDE.md)
