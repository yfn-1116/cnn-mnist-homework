from pathlib import Path
import argparse
import os
import torch

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger, log_experiment_header
from utils.visualize import plot_first_conv_layer_weights
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize first conv layer weights of trained CNN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom directory to save figure (default: use figure_dir from config)")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output figure filename (default: config filename stem)")
    parser.add_argument("--max-filters", type=int, default=64, help="Maximum number of filters to display (default: 64)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["config_name"] = args.config
    set_seed(config.get("seed", 42))

    device_str = config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    output_cfg = config["output"]
    os.makedirs(output_cfg["log_dir"], exist_ok=True)

    experiment_name = f"{Path(args.config).stem}_visualize_weights"
    logger, log_path = setup_logger(output_cfg["log_dir"], experiment_name)
    logger.info(f"Using config: {args.config}")
    log_experiment_header(logger, experiment_name, args.config, config)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Using device: {device}")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    save_dir = args.output_dir if args.output_dir is not None else output_cfg["figure_dir"]
    prefix = args.prefix if args.prefix is not None else Path(args.config).stem

    weight_vis_path = plot_first_conv_layer_weights(
        model=model,
        save_dir=save_dir,
        prefix=prefix,
        max_filters=args.max_filters,
    )

    if weight_vis_path is not None:
        logger.info(f"First conv layer weight visualization saved to: {weight_vis_path}")
    else:
        logger.warning("First conv layer weight visualization skipped (no Conv2d layer found or missing weights)")

    print(f"Visualization saved to: {weight_vis_path}")


if __name__ == "__main__":
    main()