# 仓库整理报告

## 1. 本次检查范围

本次整理重点检查了两部分：

- `outputs/figures/` 中哪些图是正式报告实际依赖，哪些只是训练或分析过程中的本地产物
- `report/report.md` 与 `report/report_for_docx_v2.md` 的图片引用策略是否一致，是否适合上传到 GitHub

同时顺带复查了仓库中不应上传到 GitHub 的内容是否已经被忽略。

## 2. 当前文件分层是否合理

当前仓库的主分层是合理的，核心原因如下：

- `configs/` 负责实验切换，避免把参数写死在代码里
- `data/`、`models/`、`engine/`、`optimizers/`、`utils/` 形成清晰的职责分层
- `outputs/` 专门存放运行产物，避免污染源码目录
- `docs/` 负责工程说明，`report/` 负责课程报告，两者定位不同

这种设计适合课程实验，因为它同时满足：

- 代码可读
- 实验可复现
- 报告可沉淀
- 后续扩展方便

## 3. 当前图片目录梳理

### 3.1 随仓库提交的正式报告图片

目前正式报告主文 `report/report.md` 依赖的图片已经统一整理到：

- `docs/assets/report/`

当前保留的正式图片为：

- `docs/assets/report/cnn_basic_long_report_loss.png`
- `docs/assets/report/cnn_basic_long_report_acc.png`
- `docs/assets/report/skip_connection_landscape.png`
- `docs/assets/report/cnn_basic_long_report_weight_dynamics.png`

这 4 张图可以视为“正式报告主文依赖图”，也是当前建议随 GitHub 提交的少量必要图片。

### 3.2 本地输出图片目录 `outputs/figures/`

当前 `outputs/figures/` 已经被收缩为两类内容：

- 正式报告相关但仍保留在本地的原始输出图
- `loss_slice_surfaces/` 下的一组分析图和说明文本

目前仍保留在 `outputs/figures/` 的主要文件包括：

- `cnn_basic_long_report_loss.png`
- `cnn_basic_long_report_acc.png`
- `cnn_basic_long_report_weight_dynamics.png`
- `skip_connection_landscape.png`
- `loss_slice_surfaces/` 下的一组损失面图片和说明文本

原先那批没有引用、命名重复的旧曲线图已经删除，例如：

- `cnn_acc.png`
- `cnn_loss.png`
- `mlp_acc.png`
- `mlp_loss.png`
- `cnn_basic_acc.png`
- `cnn_basic_loss.png`
- `cnn_skip_acc.png`
- `cnn_skip_loss.png`
- `cnn_skip_long_acc.png`
- `cnn_skip_long_loss.png`

这些图删除后，不影响正式报告与当前仓库展示。

## 4. 报告图片策略现状

### 4.1 `report/report.md`

`report/report.md` 现在已经改为引用 `../docs/assets/report/...`。

这意味着：

- 正式报告主文不再依赖被忽略的 `outputs/figures/`
- GitHub 上查看 `report/report.md` 时，正式保留的 4 张图可以正常显示
- 正式报告和仓库展示策略已经统一

### 4.2 `report/report_for_docx_v2.md`

`report/report_for_docx_v2.md` 当前引用的是 `report/assets/gradient_descent_surfaces/...` 这一路径下的图片。

这个策略更接近“导出 Word 专用版”，因为：

- 报告需要的图被收集到 `report/assets/`
- Markdown 到 Word 的相对路径更稳定
- 不直接依赖 `outputs/figures/`

但当前也存在一个现实前提：

- `report/assets/` 也被 `.gitignore` 忽略
- 所以这套图片同样不会被上传到 GitHub

因此它本质上仍然是“本地导出版”，不是“GitHub 渲染版”。

## 5. 当前最主要的问题

本次检查发现 3 个主要问题。

### 问题 1：正式报告与图片提交策略的冲突

这个问题已经修复。

当前正式报告 `report/report.md` 已改为依赖 `docs/assets/report/`，不再直接依赖被忽略的 `outputs/figures/`。

### 问题 2：存在两套并行图片体系

当前实际上有两套图片路径策略：

- `report/report.md` -> `../outputs/figures/...`
- `report/report_for_docx_v2.md` -> `report/assets/...`

这说明：

- 一套用于本地报告主文
- 一套用于 Word 导出

这种做法本身不是错，但需要明确写清楚，否则后面很容易继续混用。

### 问题 3：实验记录文件有重复段落

`report/experiment_log.md` 中存在重复记录，例如：

- `EXP-004-DEBUG` 出现两次
- `EXP-009` 出现三次

这不影响代码运行，但会影响仓库整洁度和报告可信度，建议下一轮清理时去重。

## 6. GitHub 上传内容检查结果

当前不应上传 GitHub 的内容，忽略策略基本已经到位，主要包括：

- `.idea/`
- `__pycache__/`
- `data/MNIST/`
- `outputs/checkpoints/`
- `outputs/logs/`
- `outputs/figures/`
- `outputs/predictions/`
- `report/assets/`
- `report/*.docx`
- `*.pth`、`*.pt`、`*.ckpt`
- `*.log`

当前策略可以满足“源码仓库保持轻量”这一目标。

## 7. 建议的整理原则

建议把仓库里的报告分成 3 类，并固定下来：

- `report/report.md`
  作用：仓库正式报告主文
  建议：要么改成不依赖图片，要么接受它只是本地阅读版

- `report/report_for_docx_v2.md`
  作用：Word 导出源文件
  建议：继续保留，配合 `report/assets/` 本地使用

- `report/experiment_log.md`
  作用：实验过程记录
  建议：做一次去重和格式统一

## 8. 建议的后续收口方案

当前已经采用“轻仓库 + 少量正式图片提交”的收口方案。

做法是：

- 保持 `outputs/` 和 `report/assets/` 不上传
- 将正式报告真正需要的 4 张图复制到 `docs/assets/report/`
- 让 `report/report.md` 只依赖这 4 张图

这样既保留了轻仓库策略，也保证了 GitHub 上的主报告可以正常展示。

## 9. 本次已完成的整理

本轮已经完成以下收口：

- 删除重复旧稿 `report/report_for_docx.md`
- 保留 `report/report_for_docx_v2.md` 作为当前 Word 导出源
- 新增 `report/README.md` 明确报告目录职责
- 更新 `.gitignore`，明确 `report/assets/` 为本地生成内容
- 更新 `README.md`、`docs/FILES_REFERENCE.md`、`configs/README.md`
- 补充 `report/assets/.gitkeep`
- 新建 `docs/assets/report/` 作为可提交的正式报告图片目录
- 将 `report/report.md` 的图片引用切换到 `docs/assets/report/`
- 删除 `outputs/figures/` 中一批没有引用、命名重复的旧图

## 10. 建议下一步

如果继续整理，优先级建议如下：

1. 清理 `report/experiment_log.md` 的重复记录
2. 继续统一 `report/report_for_docx_v2.md` 的本地导出图片流程说明
3. 视需要再决定是否继续精简 `loss_slice_surfaces/` 这组分析图

## 11. 结论

当前仓库的代码分层已经比较清楚，源码与产物的边界也基本建立起来了。正式报告的图片引用策略已经完成收口，现在主要剩余问题是 `report/experiment_log.md` 的重复内容，以及本地导出链路的进一步整理。

换句话说，代码层和正式报告展示层已经接近可发布状态，文档层还剩最后一次精修。
