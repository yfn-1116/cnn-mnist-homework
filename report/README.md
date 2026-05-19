# report 目录说明

本目录只保留需要长期维护的报告源文件，不保留导出的二进制文件或本地图表产物。

## 建议保留的文件

- `report.md`：仓库中的正式实验报告主文
- `report_for_docx_v2.md`：用于导出 Word 的排版版本
- `experiment_log.md`：实验记录与过程补充

## 随仓库提交的少量正式图片

- GitHub 展示所需的正式报告图片统一放在 `docs/assets/report/`
- `report/report.md` 只依赖这部分少量图片，避免直接依赖 `outputs/figures/`

## 本地生成但不提交的内容

- `assets/`：报告图片收集目录，仅本地生成，不提交到 Git
- `*.docx`：导出的 Word 文件，不提交到 Git

## 说明

- `report_for_docx_v2.md` 是当前保留的 Word 导出源，旧版重复草稿已移除。
- 如需重新收集报告图片，可运行 `python scripts/collect_report_assets.py`，脚本会把本地 `outputs/figures/` 中需要的图片复制到 `report/assets/`。
