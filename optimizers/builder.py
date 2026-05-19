import torch.optim as optim


def build_optimizer(model, config: dict):
    opt_name = config["optimizer"]["name"].lower()
    lr = config["train"]["lr"]
    weight_decay = config["train"].get("weight_decay", 0.0)

    if opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return optimizer
