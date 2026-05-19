from pathlib import Path
import torch
from tqdm import tqdm

from engine.evaluator import evaluate
from utils.checkpoint import save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False, disable=True)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return {
        "loss": avg_loss,
        "acc": avg_acc
    }


def summarize_weight_norms(model):
    total_sq_norm = 0.0
    tracked_weights = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_norm = param.detach().norm(2).item()
        total_sq_norm += param_norm ** 2
        if name.endswith("weight"):
            tracked_weights.append((name, param_norm))

    tracked_weights.sort(key=lambda item: item[0])
    first_weight_name, first_weight_norm = tracked_weights[0] if tracked_weights else ("", 0.0)
    last_weight_name, last_weight_norm = tracked_weights[-1] if tracked_weights else ("", 0.0)

    return {
        "total_weight_l2": total_sq_norm ** 0.5,
        "first_weight_name": first_weight_name,
        "first_weight_l2": first_weight_norm,
        "last_weight_name": last_weight_name,
        "last_weight_l2": last_weight_norm,
    }


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, logger):
    epochs = config["train"]["epochs"]
    checkpoint_dir = config["output"]["checkpoint_dir"]
    model_name = config["model"]["name"]
    experiment_name = Path(config.get("config_name", model_name)).stem if isinstance(config.get("config_name", model_name), str) else model_name

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "weight_l2": [],
        "first_weight_l2": [],
        "last_weight_l2": [],
    }

    best_val_acc = 0.0
    tracked_names = None

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch [{epoch}/{epochs}] started.")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        weight_stats = summarize_weight_norms(model)
        history["weight_l2"].append(weight_stats["total_weight_l2"])
        history["first_weight_l2"].append(weight_stats["first_weight_l2"])
        history["last_weight_l2"].append(weight_stats["last_weight_l2"])

        if tracked_names is None:
            tracked_names = {
                "first": weight_stats["first_weight_name"],
                "last": weight_stats["last_weight_name"],
            }
            history["tracked_weight_names"] = tracked_names

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['acc']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['acc']:.4f}, "
            f"Weight L2: {weight_stats['total_weight_l2']:.4f}"
        )
        logger.info(
            "Tracked Weights | first=%s: %.4f | last=%s: %.4f",
            weight_stats["first_weight_name"],
            weight_stats["first_weight_l2"],
            weight_stats["last_weight_name"],
            weight_stats["last_weight_l2"],
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            ckpt_name = f"{experiment_name}_best.pth"
            save_path = save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
                checkpoint_dir=checkpoint_dir,
                filename=ckpt_name
            )
            logger.info(f"Best checkpoint saved to: {save_path}")

    logger.info(f"Training finished. Best Val Acc: {best_val_acc:.4f}")
    return history, best_val_acc
