import os
import torch


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path
