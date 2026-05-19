import torch


def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total
