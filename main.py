from pathlib import Path
import argparse
import os
import torch

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger, log_experiment_header, log_experiment_footer
from utils.visualize import plot_first_conv_layer_weights, plot_training_curves

from data.loader import build_dataloaders
from models import build_model
from engine.losses import build_loss
from engine.trainer import train
from optimizers.builder import build_optimizer
from optimizers.scheduler import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="CNN/MLP for MNIST")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
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

    experiment_name = Path(args.config).stem
    logger, log_path = setup_logger(output_cfg["log_dir"], experiment_name)

    logger.info(f"Using config: {args.config}")
    log_experiment_header(logger, experiment_name, args.config, config)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(config)

    model = build_model(config).to(device)
    criterion = build_loss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history, best_val_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
    )

    plot_training_curves(
        history=history,
        save_dir=output_cfg["figure_dir"],
        prefix=experiment_name
    )
    weight_vis_path = plot_first_conv_layer_weights(
        model=model,
        save_dir=output_cfg["figure_dir"],
        prefix=experiment_name,
    )

    logger.info("Training curves saved.")
    if weight_vis_path is not None:
        logger.info(f"First conv layer weight visualization saved to: {weight_vis_path}")
    else:
        logger.info("First conv layer weight visualization skipped for current model.")

    train_loss_hist = history.get("train_loss", history.get("loss", []))
    train_acc_hist = history.get("train_acc", history.get("acc", []))
    val_loss_hist = history.get("val_loss", [])
    val_acc_hist = history.get("val_acc", [])

    log_experiment_footer(
        logger,
        experiment_name,
        best_val_acc,
        final_train_loss=train_loss_hist[-1] if train_loss_hist else None,
        final_train_acc=train_acc_hist[-1] if train_acc_hist else None,
        final_val_loss=val_loss_hist[-1] if val_loss_hist else None,
        final_val_acc=val_acc_hist[-1] if val_acc_hist else None,
    )

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
