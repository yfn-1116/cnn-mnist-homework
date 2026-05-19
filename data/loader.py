from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from data.transforms import build_transforms


def build_dataloaders(config: dict):
    """
    构建 train / val / test dataloaders
    若本地已有 MNIST 原始文件，则不会重复下载；
    若缺失，则会自动补齐并完成处理。
    """
    data_cfg = config["data"]
    root = data_cfg["root"]

    train_transform, test_transform = build_transforms(
        normalize=data_cfg.get("normalize", True),
        augmentation=data_cfg.get("augmentation", False)
    )

    train_dataset_full = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=test_transform
    )

    val_size = 5000
    train_size = len(train_dataset_full) - val_size

    train_subset, val_subset = random_split(
        train_dataset_full,
        [train_size, val_size]
    )

    val_dataset_full = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=test_transform
    )

    train_dataset = Subset(train_dataset_full, train_subset.indices)
    val_dataset = Subset(val_dataset_full, val_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
