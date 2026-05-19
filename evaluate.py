from pathlib import Path
import argparse
import os
import torch

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import setup_logger, log_experiment_header
from data.loader import build_dataloaders
from models import build_model
from engine.losses import build_loss
from engine.evaluator import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on MNIST")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate")
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

    experiment_name = f"{Path(args.config).stem}_eval_{args.split}"
    logger, log_path = setup_logger(output_cfg["log_dir"], experiment_name)
    logger.info(f"Using config: {args.config}")
    log_experiment_header(logger, experiment_name, args.config, config)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(config)
    dataloader = val_loader if args.split == "val" else test_loader

    model = build_model(config).to(device)
    criterion = build_loss()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device
    )

    logger.info("-" * 60)
    logger.info("Evaluation split: %s", args.split)
    logger.info("Loss: %.4f", metrics["loss"])
    logger.info("Accuracy: %.4f", metrics["acc"])
    logger.info("-" * 60)

    print(f"{args.split} loss: {metrics['loss']:.4f}")
    print(f"{args.split} acc: {metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
