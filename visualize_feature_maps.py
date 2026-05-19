from pathlib import Path
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from utils.config import load_config
from utils.logger import setup_logger, log_experiment_header
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize feature maps after the first convolutional layer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom directory to save figure")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output figure filename")
    parser.add_argument("--max-maps", type=int, default=32, help="Maximum number of feature maps to display")
    return parser.parse_args()


def build_transform(normalize: bool = True):
    t = [transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()]
    if normalize:
        t.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(t)


def get_first_feature_layer(model):
    seen_first_conv = False

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not seen_first_conv:
            seen_first_conv = True
            continue

        if seen_first_conv and isinstance(module, nn.ReLU):
            return name, module

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            return name, module

    return None, None


def normalize_feature_maps(feature_maps: np.ndarray):
    normalized = []

    for feature_map in feature_maps:
        fmap_min = feature_map.min()
        fmap_max = feature_map.max()

        if fmap_max > fmap_min:
            normalized_map = (feature_map - fmap_min) / (fmap_max - fmap_min)
        else:
            normalized_map = np.zeros_like(feature_map)

        normalized.append(normalized_map)

    return np.array(normalized)


def plot_feature_maps(feature_maps: np.ndarray, save_path: str, layer_name: str):
    num_maps = feature_maps.shape[0]
    cols = min(8, num_maps)
    rows = math.ceil(num_maps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 1.9))
    axes = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= num_maps:
            continue
        ax.imshow(feature_maps[idx], cmap="gray")
        ax.set_title(f"FM{idx + 1}", fontsize=8)

    fig.suptitle(f"Feature Maps after First Activation Layer ({layer_name}, {num_maps} channels)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    config["config_name"] = args.config

    device_str = config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    output_cfg = config["output"]
    os.makedirs(output_cfg["log_dir"], exist_ok=True)

    experiment_name = f"{Path(args.config).stem}_feature_maps"
    logger, log_path = setup_logger(output_cfg["log_dir"], experiment_name)
    logger.info(f"Using config: {args.config}")
    log_experiment_header(logger, experiment_name, args.config, config)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Using device: {device}")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    layer_name, target_layer = get_first_feature_layer(model)
    if target_layer is None:
        raise ValueError("No suitable feature layer found in the model")

    activation = {}

    def hook_fn(_module, _inputs, output):
        activation["feature_maps"] = output.detach().cpu()

    handle = target_layer.register_forward_hook(hook_fn)

    transform = build_transform(config["data"].get("normalize", True))
    image = Image.open(args.image).convert("L")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x)

    handle.remove()

    if "feature_maps" not in activation:
        raise RuntimeError("Failed to capture feature maps from the first convolutional layer")

    feature_maps = activation["feature_maps"][0].numpy()
    num_maps = min(feature_maps.shape[0], args.max_maps)
    feature_maps = normalize_feature_maps(feature_maps[:num_maps])

    save_dir = args.output_dir if args.output_dir is not None else output_cfg["figure_dir"]
    os.makedirs(save_dir, exist_ok=True)
    prefix = args.prefix if args.prefix is not None else Path(args.image).stem
    save_path = os.path.join(save_dir, f"{prefix}_feature_maps.png")

    plot_feature_maps(feature_maps, save_path, layer_name)

    logger.info(f"Feature map visualization saved to: {save_path}")
    print(f"Visualization saved to: {save_path}")


if __name__ == "__main__":
    main()
