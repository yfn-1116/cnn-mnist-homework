from pathlib import Path
import argparse
import os
import torch
from PIL import Image
from torchvision import transforms

from utils.config import load_config
from utils.logger import setup_logger, log_experiment_header
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for MNIST model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    return parser.parse_args()
def build_transform(normalize: bool = True):
    t = [transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()]
    if normalize:
        t.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(t)


def main():
    args = parse_args()
    config = load_config(args.config)
    config["config_name"] = args.config

    device_str = config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    output_cfg = config["output"]
    os.makedirs(output_cfg["log_dir"], exist_ok=True)
    os.makedirs(output_cfg["prediction_dir"], exist_ok=True)

    experiment_name = f"{Path(args.config).stem}_infer"
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

    transform = build_transform(config["data"].get("normalize", True))
    image = Image.open(args.image).convert("L")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    logger.info("-" * 60)
    logger.info("Predicted digit: %d", pred)
    logger.info("Confidence: %.4f", conf)
    logger.info("-" * 60)

    out_path = os.path.join(output_cfg["prediction_dir"], f"{Path(args.image).stem}_pred.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"image: {args.image}\n")
        f.write(f"predicted_digit: {pred}\n")
        f.write(f"confidence: {conf:.4f}\n")

    print(f"predicted digit: {pred}")
    print(f"confidence: {conf:.4f}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
