from pathlib import Path
import re
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualize import (
    generate_skip_connection_figure,
    generate_skip_connection_figure_3d,
    plot_optimization_path_3d,
)

OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
REPORT_ASSETS = PROJECT_ROOT / "report" / "assets"

CONFIG_TO_LOG = {
    "mlp_baseline": PROJECT_ROOT / "outputs" / "logs" / "mlp_train.log",
    "cnn_basic": PROJECT_ROOT / "outputs" / "logs" / "cnn_basic.log",
    "cnn_bn_adamw": PROJECT_ROOT / "outputs" / "logs" / "cnn_train.log",
    "cnn_aug": PROJECT_ROOT / "outputs" / "logs" / "cnn_train.log",
    "cnn_reg": PROJECT_ROOT / "outputs" / "logs" / "cnn_train.log",
    "cnn_aug_tuned": PROJECT_ROOT / "outputs" / "logs" / "cnn_train.log",
    "cnn_basic_long": PROJECT_ROOT / "outputs" / "logs" / "cnn_train.log",
    "cnn_skip": PROJECT_ROOT / "outputs" / "logs" / "cnn_skip.log",
    "cnn_skip_long": PROJECT_ROOT / "outputs" / "logs" / "cnn_skip_long.log",
}

EPOCH_METRIC_RE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\] "
    r"Train Loss: (?P<train_loss>\d+\.\d+), "
    r"Train Acc: (?P<train_acc>\d+\.\d+), "
    r"Val Loss: (?P<val_loss>\d+\.\d+), "
    r"Val Acc: (?P<val_acc>\d+\.\d+)"
)


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


def parse_latest_run(log_path: Path, config_stem: str):
    if not log_path.exists():
        return None

    marker = f"Using config: configs/{config_stem}.yaml"
    current = None
    matched_runs = []

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if "Using config:" in line:
            if current and current["config"] == config_stem and current["train_loss"]:
                matched_runs.append(current)

            current = {"config": None, "train_loss": [], "val_loss": [], "val_acc": []}
            if marker in line:
                current["config"] = config_stem
            continue

        if current is None or current["config"] != config_stem:
            continue

        match = EPOCH_METRIC_RE.search(line)
        if match:
            current["train_loss"].append(float(match.group("train_loss")))
            current["val_loss"].append(float(match.group("val_loss")))
            current["val_acc"].append(float(match.group("val_acc")))

    if current and current["config"] == config_stem and current["train_loss"]:
        matched_runs.append(current)

    return matched_runs[-1] if matched_runs else None


def main():
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)

    copy_targets = [
        "mlp_acc.png",
        "mlp_loss.png",
        "cnn_basic_acc.png",
        "cnn_basic_loss.png",
        "cnn_basic_long_report_acc.png",
        "cnn_basic_long_report_loss.png",
        "cnn_basic_long_report_weight_dynamics.png",
        "cnn_skip_acc.png",
        "cnn_skip_loss.png",
        "cnn_skip_long_acc.png",
        "cnn_skip_long_loss.png",
    ]

    for name in copy_targets:
        copy_if_exists(OUTPUT_FIGURES / name, REPORT_ASSETS / name)

    for config_stem, log_path in CONFIG_TO_LOG.items():
        history = parse_latest_run(log_path, config_stem)
        if history is None:
            continue

        plot_optimization_path_3d(
            history=history,
            save_dir=str(REPORT_ASSETS),
            prefix=config_stem,
            title=f"{config_stem} Optimization Path",
        )

    fig_2d = generate_skip_connection_figure(str(REPORT_ASSETS), "skip_connection_landscape.png")
    fig_3d = generate_skip_connection_figure_3d(str(REPORT_ASSETS), "skip_connection_landscape_3d.png")

    print(f"report assets ready: {REPORT_ASSETS}")
    print(f"2d skip figure: {fig_2d}")
    print(f"3d skip figure: {fig_3d}")


if __name__ == "__main__":
    main()
