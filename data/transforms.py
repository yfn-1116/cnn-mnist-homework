from torchvision import transforms


def build_transforms(normalize: bool = True, augmentation: bool = False):
    """
    构建训练集和测试集的数据预处理流程
    """
    mean = (0.1307,)
    std = (0.3081,)

    train_transforms = []

    if augmentation:
        train_transforms.extend([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    train_transforms.append(transforms.ToTensor())

    if normalize:
        train_transforms.append(transforms.Normalize(mean, std))

    test_transforms = [transforms.ToTensor()]
    if normalize:
        test_transforms.append(transforms.Normalize(mean, std))

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    return train_transform, test_transform
