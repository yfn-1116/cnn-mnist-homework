# 深度学习实验报告

## 作业项目

基于 CNN 的 MNIST 手写数字识别实验

## 实验目的及要求

在上一份《基于多层感知机的 MNIST 手写数字识别》实验中，已经完成了基于 MLP 的手写数字分类任务，最高准确率约为 97%。该实验说明了全连接神经网络能够完成基本分类，但也暴露出其在图像任务中的明显不足：输入图像在进入模型前被直接展平，空间结构信息被破坏，模型难以充分利用局部像素之间的关联关系。

本次实验在上一份作业基础上继续展开，实验主题转为“基于 CNN 的 MNIST 手写数字识别实验”。实验要求不再局限于实现单一模型，而是要参考课程给出的方法表，在重新完成 MNIST 手写数字识别任务的同时，尽可能将课程中涉及的常见方法纳入统一实验框架中，并通过对照实验分析各方法的实际作用。

本次实验的主要目标包括：

1. 使用 CNN 替代 MLP，验证卷积神经网络在图像分类任务中的结构优势。
2. 在统一项目框架中纳入 Adam / AdamW、Dropout、Weight Decay、Initialization、BatchNorm / Normalization、Data Augmentation、Skip Connection 等方法。
3. 除了完成训练，还需要建立规范的工程结构，实现训练、评估、推理三条完整链路。
4. 通过配置驱动方式管理实验参数，使实验过程可复现、可扩展、可比较。
5. 形成适合课程作业提交和后续上传 GitHub 的项目组织方式，包括代码、文档、实验记录和实验报告。

因此，本次实验既是对卷积神经网络原理和实现的验证，也是对实验工程组织能力的一次系统训练。

## 实验设备环境

本实验在 Python 与 PyTorch 环境下完成，主要依赖如下：

- Python 3
- PyTorch
- torchvision
- matplotlib
- numpy
- pyyaml
- tqdm
- scikit-learn

项目采用模块化分层设计，主要目录包括：

- `configs/`：管理实验配置文件
- `data/`：负责数据读取与预处理
- `models/`：负责模型定义，包括 MLP baseline、基础 CNN 和带残差结构的 CNN
- `engine/`：负责训练与评估流程
- `optimizers/`：负责优化器与调度器构建
- `utils/`：负责日志、checkpoint、配置读取、可视化等通用功能
- `docs/`：负责项目说明文档
- `report/`：负责实验报告与实验记录

项目已经完成以下三条主链路：

- `main.py`：训练主链路
- `evaluate.py`：评估主链路
- `infer.py`：推理主链路

日志文件按实验名保存，checkpoint 按实验名保存，推理结果保存到 `outputs/predictions/`。整个项目采用配置驱动方式管理实验，关键配置项包括：

- `seed`
- `device`
- `data.root`
- `data.batch_size`
- `data.normalize`
- `data.augmentation`
- `model.name`
- `model.dropout`
- `model.use_batchnorm`
- `model.init_method`
- `train.epochs`
- `train.lr`
- `train.weight_decay`
- `optimizer.name`
- `scheduler.use`
- `output.*`

这种设计使得本项目不再是单一训练脚本，而是一个具备实验管理能力、适合持续扩展的课程工程项目。

## 1. 实验内容与实现

本次实验是在上一份 MLP 报告基础上发展而来的 CNN 项目，因此实验内容既包括模型层面的升级，也包括工程实现层面的完善。

首先，在模型层面保留了 `MLP baseline`，作为对照组使用。这样可以清楚观察从 MLP 切换到 CNN 后性能变化的幅度，明确本次实验中最重要的结构性改进到底来自哪里。

其次，构建了基础 CNN 作为本次实验主模型。在此基础上，再逐步引入课程中常见的若干方法，包括：

- Adam / AdamW
- Dropout
- Weight Decay
- He Initialization
- BatchNorm
- Data Augmentation
- Skip Connection

这些方法不是一次性全部加入，而是采用逐组递进方式进行比较，以便分析每一步参数与结构调整分别带来了什么影响。

再次，在工程实现层面，本项目采用配置驱动方式组织实验。每组实验的参数不再直接写死在代码中，而是通过 `configs/*.yaml` 管理。训练、评估、推理三条链路分别通过不同入口脚本完成，实验结果记录在日志中，模型权重以实验名保存，实验报告和实验记录独立存放。这种组织方式使得实验过程更加规范，也便于后续复现实验与扩展实验。

最后，在文档管理方面，本项目将说明文档集中放置在 `docs/` 中，将正式报告与实验记录分开管理，使项目整体结构更适合后续整理与提交。

## 2. 模型设计

### 2.1 MLP baseline 的作用

MLP baseline 的结构与上一份实验报告保持衔接，它将输入图像展平后，通过全连接层完成分类。该模型的优点在于实现简单，能够作为课程实验的基础起点；但由于它无法利用图像的局部空间结构，因此在图像任务中的表达能力有限。

本次实验保留 MLP baseline 的目的，并不是继续优化 MLP 本身，而是为 CNN 提供一个可直接比较的参照系。通过保留这个对照组，可以更清楚地说明 CNN 到底带来了多大幅度的性能提升。

### 2.2 基础 CNN 模型

本实验的主模型为基础 CNN。其整体结构可以概括为两部分：

1. 特征提取部分：由卷积层组成，用于提取图像局部特征。
2. 分类部分：将卷积特征图展平，并通过全连接层输出 10 类分类结果。

与 MLP 相比，CNN 最重要的优势在于：

- 通过局部感受野提取边缘、笔画和局部形状等图像特征；
- 通过参数共享减少参数量，提高学习效率；
- 保留二维图像结构，更适合处理图像输入。

因此，从理论上讲，将 MLP 替换为 CNN 应当带来明显提升，这也是本实验需要首先验证的核心问题。

### 2.3 引入训练技巧与结构改进

在基础 CNN 建立之后，本实验进一步围绕课程中的常见方法进行扩展：

- 使用 `AdamW` 代替 `Adam`，观察优化器变化对结果的影响；
- 加入 `BatchNorm`，观察归一化是否提高训练稳定性；
- 加入 `Dropout` 和 `Weight Decay`，观察正则化效果；
- 启用 `Data Augmentation`，观察样本增强是否改善泛化；
- 引入 `Skip Connection`，观察残差连接是否在当前任务中带来优化优势。

这些方法的共同特点是：它们都建立在基础 CNN 之上，属于“在已有主模型之上继续优化”的路径。因此，它们的实验结果可以帮助判断：在 MNIST 任务上，究竟是模型结构本身更重要，还是训练技巧叠加更重要。

## 3. 实验过程

本次实验按递进主线依次完成，每一组实验都相对上一组只改变少量关键因素，这样有利于清楚分析参数逐步变化带来的效果变化。

### 3.1 MLP baseline

首先运行 `mlp_baseline`，作为整个实验主线的起点。其结果为：

- Best Validation Accuracy: `0.9786`

这一结果与上一份作业保持衔接，说明 MLP 已经可以较好地完成基本分类任务，但仍存在明显上限。由于 MLP 不擅长处理图像空间结构，因此它更适合作为对照组，而不是最终方案。

![mlp_baseline_gradient_descent_surface](assets/gradient_descent_surfaces/mlp_baseline_gradient_descent_surface.png)

图 3-1  MLP baseline 实验的梯度下降示意图。图中用三维损失曲面表示参数空间中的优化地形，用黑色折线表示参数从初始点逐步下降到低点附近的过程。

### 3.2 CNN basic

第二组实验将模型从 MLP 替换为基础 CNN，实验设置如下：

- optimizer: `adam`
- lr: `0.001`
- weight_decay: `0.0`
- batch_size: `64`
- dropout: `0.3`
- batchnorm: `False`
- augmentation: `False`
- epochs: `10`

实验结果如下：

- Best Val Acc: `0.9918`
- Final Train Loss: `0.0157`
- Final Train Acc: `0.9949`
- Final Val Loss: `0.0321`
- Final Val Acc: `0.9918`

这一步带来了本次项目中最显著的性能提升。相比 `mlp_baseline` 的 `0.9786`，`cnn_basic` 达到 `0.9918`，说明“从 MLP 到 CNN”的结构性改动是最关键的。卷积结构能够更好地利用图像局部特征，这一结果与理论预期完全一致。

此外，从最终训练集与验证集结果来看，训练准确率 `0.9949`、验证准确率 `0.9918`，两者差距并不大，说明模型虽然拟合充分，但没有出现严重过拟合。这也是基础 CNN 能够成为后续所有实验主线基础的原因。

![cnn_basic_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_basic_gradient_descent_surface.png)

图 3-2  基础 CNN 实验的梯度下降示意图。可以看到路径下降更直接，说明该结构在当前任务上的优化过程较为顺畅。

### 3.3 CNN + BatchNorm + AdamW

第三组实验在基础 CNN 上加入 `BatchNorm`，并将优化器改为 `AdamW`，主要考虑有两点：

1. BatchNorm 可能改善训练稳定性；
2. AdamW 可能带来更好的参数更新与泛化能力。

实验结果如下：

- optimizer: `AdamW`
- epochs: `10`
- Best Val Acc: `0.9890`

这一结果没有达到预期。相比 `cnn_basic` 的 `0.9918`，该组实验反而略低。说明在当前任务、当前网络深度和当前参数设置下，BatchNorm 与 AdamW 的引入并没有进一步提升性能。其合理解释是：MNIST 任务较简单，基础 CNN 已经比较容易训练，BatchNorm 的稳定训练优势并未充分体现；与此同时，Adam 也已经能够较好地完成当前优化任务，因此 AdamW 没有形成明显优势。

![cnn_bn_adamw_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_bn_adamw_gradient_descent_surface.png)

图 3-3  CNN + BatchNorm + AdamW 实验的梯度下降示意图。该图主要用于教学展示其优化路径特征，并辅助说明该组合在本实验中的性能并未超过基础 CNN。

### 3.4 CNN + Data Augmentation

第四组实验在 CNN 上启用数据增强，预期作用是通过增加训练样本的多样性来改善模型泛化能力。实验中采用的增强方式主要是随机旋转和随机平移。

初次运行时，虽然配置中已经开启 augmentation，但实验结果与未增强实验几乎一致，这显然不符合预期。随后对数据加载逻辑进行排查，发现训练集和验证集共享了同一底层数据集对象，导致训练增强变换被验证集变换覆盖，从而使增强实际上没有生效。

修复该问题后重新训练，结果为：

- epochs: `10`
- Best Val Acc: `0.9910`

这一结果高于 `cnn_bn_adamw` 的 `0.9890`，说明数据增强在修复后确实生效，并带来了正向作用；但它仍略低于 `cnn_basic` 的 `0.9918`。因此，这一组实验部分达到预期：增强是有价值的，但在当前增强强度和超参数条件下，并没有超过基础 CNN。

![cnn_aug_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_aug_gradient_descent_surface.png)

图 3-4  CNN + Data Augmentation 实验的梯度下降示意图。相较于基础 CNN，该图体现了加入增强后优化路径仍能稳定下降，但整体收益有限。

### 3.5 CNN + Dropout + Weight Decay

第五组实验在 CNN 中加入 `Dropout` 和 `Weight Decay`，其预期作用是通过正则化缓解过拟合，提高模型泛化能力。实验结果为：

- optimizer: `Adam`
- epochs: `10`
- Best Val Acc: `0.9902`

这一结果高于 `cnn_bn_adamw` 的 `0.9890`，说明正则化确实带来一定帮助；但仍低于 `cnn_basic` 的 `0.9918`，也略低于 `cnn_aug` 的 `0.9910`。这说明在当前任务中，基础 CNN 本身已经有较好的泛化能力，进一步加入正则化虽然有实验价值，但收益有限，并未形成显著突破。

![cnn_reg_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_reg_gradient_descent_surface.png)

图 3-5  CNN + Dropout + Weight Decay 实验的梯度下降示意图。该图用于辅助说明正则化后的优化轨迹更平稳，但最终性能提升并不显著。

### 3.6 调参版 CNN + Data Augmentation

第六组实验在增强版 CNN 基础上继续调参，目的在于观察超参数变化是否能让增强实验更进一步。实验结果如下：

- optimizer: `AdamW`
- epochs: `10`
- Best Val Acc: `0.9890`

该结果并没有超过 `cnn_aug`，也未超过 `cnn_basic`。说明当前这轮调参没有达到预期。其解释是：数据增强策略本身虽然有效，但和优化器、学习率、正则化强度之间存在耦合关系，如果调整方向不合适，反而可能抵消原有收益。

![cnn_aug_tuned_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_aug_tuned_gradient_descent_surface.png)

图 3-6  调参版 CNN + Data Augmentation 实验的梯度下降示意图。图中展示了调参后仍可收敛，但下降路径没有带来更优的最终结果。

### 3.7 CNN basic long training

由于前面统一采用 `10` 个 epoch 主要是为了公平横向比较，但不同模型和方法的收敛速度并不完全相同，因此第七组实验对表现较好的基础 CNN 进行了更长训练。实验结果如下：

- optimizer: `Adam`
- epochs: `15`
- Best Val Acc: `0.9924`

这一结果超过了 `cnn_basic` 的 `0.9918`，成为新的最优验证结果。说明基础 CNN 在 10 个 epoch 时并未完全训练充分，延长训练轮数能够继续带来收益。这一组实验的意义非常大，因为它说明：在当前任务中，与其继续堆叠复杂技巧，不如对基础 CNN 进行更充分训练，效果更直接、更稳定。

![cnn_basic_long_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_basic_long_gradient_descent_surface.png)

图 3-7  基础 CNN 长训练实验的梯度下降示意图。该图体现了更充分训练时，参数路径能够继续向更低区域推进，这与其最佳验证准确率最高的结果一致。

### 3.8 CNN + Skip Connection

第八组实验在基础 CNN 上加入 Skip Connection，用于验证残差连接在当前网络结构中的作用。实验结果如下：

- optimizer: `Adam`
- epochs: `10`
- Best Val Acc: `0.9910`

该结果与 `cnn_aug` 相同，但仍低于 `cnn_basic`。这说明在当前较浅网络结构下，Skip Connection 的优势并不明显。其原因在于，跳连结构更常用于缓解深层网络中的梯度传播困难，而本实验模型整体较浅，这类问题并不突出，因此残差连接没有形成显著性能优势。

![cnn_skip_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_skip_gradient_descent_surface.png)

图 3-8  CNN + Skip Connection 实验的梯度下降示意图。该图用于说明引入跳连后优化路径较平滑，但在当前浅层网络中并未转化为明显的精度优势。

### 3.9 CNN + Skip Connection long training

第九组实验对带 Skip Connection 的模型继续训练到 `15` 个 epoch，实验结果如下：

- optimizer: `Adam`
- epochs: `15`
- Best Val Acc: `0.9916`

这一结果高于 `cnn_skip` 的 `0.9910`，说明长训练对 Skip 模型同样有帮助；但它仍然没有超过 `cnn_basic_long` 的 `0.9924`。这进一步说明，在当前较浅网络结构和任务规模下，Skip Connection 并不是主要性能提升来源。

![cnn_skip_long_gradient_descent_surface](assets/gradient_descent_surfaces/cnn_skip_long_gradient_descent_surface.png)

图 3-9  CNN + Skip Connection 长训练实验的梯度下降示意图。相比短训练版本，路径继续向低点靠近，但最终效果仍未超过基础 CNN 长训练方案。

### 3.10 最优模型测试集评估

在完成验证集对比实验后，选取当前表现最优的 `cnn_basic_long` 模型，在测试集上进行正式评估。结果如下：

- config: `cnn_basic_long`
- Test Loss: `0.0344`
- Test Accuracy: `0.9928`

这一结果说明当前最优模型不仅在验证集上表现最好，也在独立测试集上取得了稳定而较高的分类性能，因此可以作为本次实验的最终最优方案。

### 3.11 单张样例图片推理验证

在完成训练和测试评估之后，进一步使用 `cnn_basic_long` 模型对导出的 MNIST 测试样例图片进行推理验证。结果如下：

- 使用模型：`cnn_basic_long`
- 预测结果：`7`
- Confidence: `1.0000`

这一结果说明项目的推理链路已经打通，模型能够完成单张图片的预处理、前向推理和结果输出，验证了项目在训练、评估之外还具备实际推理能力。

## 4. 实验结果

### 4.1 实验结果汇总表

| 实验名称 | 主要改动 | Epochs | Best Validation Accuracy |
|---|---|---:|---:|
| mlp_baseline | MLP 基线 | 10 | 0.9786 |
| cnn_basic | MLP → CNN，Adam，dropout=0.3 | 10 | 0.9918 |
| cnn_bn_adamw | CNN + BatchNorm + AdamW | 10 | 0.9890 |
| cnn_aug | CNN + Data Augmentation（修复增强未生效问题后） | 10 | 0.9910 |
| cnn_reg | CNN + Dropout + Weight Decay | 10 | 0.9902 |
| cnn_aug_tuned | 调参版 CNN + Data Augmentation | 10 | 0.9890 |
| cnn_basic_long | 基础 CNN 延长训练 | 15 | 0.9924 |
| cnn_skip | CNN + Skip Connection | 10 | 0.9910 |
| cnn_skip_long | CNN + Skip Connection 延长训练 | 15 | 0.9916 |

![all_experiments_gradient_descent_surface](assets/gradient_descent_surfaces/all_experiments_gradient_descent_surface.png)

图 4-1  各实验梯度下降路径的综合对比图。图中将不同实验的下降轨迹叠加在同一张三维损失曲面上，并使用不同颜色区分，以便直观比较不同实验在教学示意意义下的优化路径差异。

### 4.2 最优模型测试结果

当前最优模型为 `cnn_basic_long`，测试集结果如下：

- Test Loss: `0.0344`
- Test Accuracy: `0.9928`

### 4.3 单样例推理结果

使用 `cnn_basic_long` 模型对导出的 MNIST 测试样例图进行推理，结果如下：

- 预测类别：`7`
- Confidence: `1.0000`

## 5. 结果分析与讨论

从整个实验主线来看，本次项目最显著的性能提升来自 `MLP → CNN`。这一组改动使验证准确率从 `0.9786` 提升到 `0.9918`，是所有实验中最关键的结构性改进。这说明在图像任务中，卷积结构对局部空间特征的提取能力远强于直接展平输入的 MLP。

相比之下，后续加入的 BatchNorm、AdamW、Dropout、Weight Decay、Data Augmentation 和 Skip Connection 虽然都具有实验价值，但在当前任务和参数设置下都没有超过 `cnn_basic_long`。这说明在 MNIST 这样相对简单的数据集上，模型结构本身的改进比附加技巧的堆叠更重要。

从实验公平性角度看，前期大多数实验统一设置为 `10` 个 epoch，主要是为了保证横向比较条件一致。但不能忽略的一点是：不同方法的收敛速度本来就可能不同，仅用统一轮数比较并不能完全代表其最终潜力。因此，后续又对表现较好的模型进行了更长训练。最终 `cnn_basic_long` 达到 `0.9924`，说明在当前任务上，“基础 CNN + 更充分训练”比“基础 CNN + 更复杂技巧”更有效。

另一个重要经验来自数据增强实验。最开始出现了“配置已经开启 augmentation，但结果与未增强实验几乎一致”的现象，后来通过排查数据加载逻辑，发现训练增强实际上没有真正生效。修复后，`cnn_aug` 提升到 `0.9910`。这说明在深度学习实验中，不能只看表面配置是否写对，还必须检查数据流和实现是否真正起作用。否则就可能错误判断某种方法“没有效果”。

Skip Connection 的实验结果也值得分析。`cnn_skip` 为 `0.9910`，`cnn_skip_long` 为 `0.9916`，即使训练到 `15` 个 epoch 仍然没有超过 `cnn_basic_long`。这一结果并不说明残差连接没有价值，而是说明在当前较浅网络结构中，其优势不明显。残差连接通常更适合较深网络，因为它主要解决深层网络训练中的优化困难；而本实验的 CNN 规模较小，这类问题本身并不突出，因此跳连没有体现出决定性优势。

此外，分析实验结果时不能只看 accuracy，还应结合训练过程中的 train loss、val loss、train/val acc 差距以及是否训练充分等综合判断。例如 `cnn_basic` 的最终结果为：

- Final Train Loss: `0.0157`
- Final Train Acc: `0.9949`
- Final Val Loss: `0.0321`
- Final Val Acc: `0.9918`

从这些数值可以看出，训练集和验证集之间存在一定差距，但差距不大，说明模型在 10 个 epoch 时已经训练较充分，同时未出现严重过拟合。进一步把 epoch 增加到 `15` 后，模型最佳验证准确率提升到 `0.9924`，说明后续收益主要来自“训练更充分”，而不是模型本身存在明显缺陷。

### 5.1 方法覆盖情况

本项目已在统一实验框架下实现并验证了以下方法：

- CNN
- Adam / AdamW
- Dropout
- Weight Decay
- Initialization（He Initialization）
- Normalization / BatchNorm
- Data Augmentation
- Skip Connection
- 使用 CrossEntropyLoss 而非 accuracy 作为 loss

同时，本项目优先完成了主线实验，并为后续扩展预留了接口。尚未完整展开但已经明确保留为后续方向的方法包括：

- More training data
- Semi-supervised
- 更复杂的 Parameter Regularization

这些方向并不是简单地“未完成”，而是本次作业阶段优先聚焦于主线实验：先完成 MLP 到 CNN 的结构升级，再围绕若干核心训练技巧进行系统比较；在此基础上，为后续继续扩展更复杂实验保留了清晰接口和工程结构。

## 6. 实验结论

本次实验是在上一份 MLP for MNIST 报告基础上继续展开的 CNN 项目。通过系统完成 MLP baseline、基础 CNN、BatchNorm、AdamW、Dropout、Weight Decay、Data Augmentation、Skip Connection、长训练、测试集评估和单样例推理验证，可以得到如下结论。

首先，本次实验中最关键的提升来自于从 MLP 切换到 CNN。这一步使验证准确率从 `0.9786` 显著提升到 `0.9918`，说明在图像任务中，卷积结构带来的局部特征提取能力是决定性优势。

其次，BatchNorm、AdamW、Dropout、Weight Decay、Data Augmentation 和 Skip Connection 虽然都具有实验价值，但在当前任务和参数设置下均未超过 `cnn_basic_long`。这说明在 MNIST 这样相对简单的任务中，继续堆叠复杂技巧并不一定比“基础 CNN 训练更充分”更有效。

再次，统一 `10` 个 epoch 的设置主要用于公平横向比较，但不同方法的收敛速度并不相同，因此后续对表现较好的模型进行更长训练是合理且必要的。最终 `cnn_basic_long` 取得 `0.9924` 的最佳验证准确率，并在测试集上达到 `0.9928`，成为本项目的最终最优方案。

同时，本实验还完成了训练、评估、推理三条主链路，建立了配置驱动的实验管理方式，实现了日志、checkpoint、推理结果和文档的规范组织，使项目具备了较完整的工程化结构。

综合来看，本次实验的主要成果可以概括为以下几点：

1. 完成了从 MLP 到 CNN 的项目升级，并验证了 CNN 在图像任务中的显著优势。
2. 在统一实验框架下系统比较了多种课程方法，获得了一组真实、清晰、可解释的对照结果。
3. 对训练过程、参数变化、数据增强问题修复、长训练收益和 Skip Connection 作用进行了较完整的过程分析。
4. 在工程实现层面完成了训练、评估、推理、配置管理、文档组织等完整链路。

通过本次实验，对卷积神经网络的结构优势、训练技巧的真实作用、实验实现正确性的重要性以及深度学习项目工程化组织方式都有了更深入的理解。实验也存在一定局限，例如网络规模仍较浅、部分方法尚未在更大训练轮数或更复杂参数设置下充分展开，但这些内容已经在当前项目中保留了后续扩展接口。总体而言，本次实验已经较完整地实现了基于 CNN 的 MNIST 手写数字识别主线任务，并为后续更深入的深度学习实验奠定了基础。
