import logging
import os
import sys


def setup_logger(log_dir: str, experiment_name: str = "experiment"):
    os.makedirs(log_dir, exist_ok=True)

    logger_name = f"mnist_logger_{experiment_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def log_experiment_header(logger, experiment_name: str, config_path: str, config: dict):
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    optimizer_cfg = config.get("optimizer", {})

    logger.info("=" * 60)
    logger.info("Experiment: %s", experiment_name)
    logger.info("Config: %s", config_path)
    logger.info(
        "Settings | optimizer=%s | lr=%s | weight_decay=%s | batch_size=%s | dropout=%s | batchnorm=%s | augmentation=%s | epochs=%s",
        optimizer_cfg.get("name", "unknown"),
        train_cfg.get("lr", "unknown"),
        train_cfg.get("weight_decay", "unknown"),
        data_cfg.get("batch_size", "unknown"),
        model_cfg.get("dropout", "unknown"),
        model_cfg.get("use_batchnorm", "unknown"),
        data_cfg.get("augmentation", "unknown"),
        train_cfg.get("epochs", "unknown"),
    )
    logger.info("=" * 60)


def log_experiment_footer(
    logger,
    experiment_name: str,
    best_val_acc,
    final_train_loss=None,
    final_train_acc=None,
    final_val_loss=None,
    final_val_acc=None,
):
    logger.info("-" * 60)
    logger.info("Experiment Finished: %s", experiment_name)
    logger.info("Best Val Acc: %.4f", best_val_acc if best_val_acc is not None else -1.0)

    if final_train_loss is not None:
        logger.info("Final Train Loss: %.4f", final_train_loss)
    if final_train_acc is not None:
        logger.info("Final Train Acc: %.4f", final_train_acc)
    if final_val_loss is not None:
        logger.info("Final Val Loss: %.4f", final_val_loss)
    if final_val_acc is not None:
        logger.info("Final Val Acc: %.4f", final_val_acc)

    logger.info("-" * 60)
