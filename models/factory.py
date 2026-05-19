from models.cnn import CNN
from models.cnn_residual import ResidualCNN
from models.initialize import initialize_weights
from models.mlp import MLP


def build_model(config: dict):
    model_cfg = config["model"]
    model_name = model_cfg["name"].lower()

    if model_name == "mlp":
        model = MLP(
            input_dim=model_cfg.get("input_dim", 784),
            hidden_dims=model_cfg.get("hidden_dims", [256, 128]),
            num_classes=model_cfg.get("num_classes", 10),
            dropout=model_cfg.get("dropout", 0.0),
        )
    elif model_name == "cnn":
        model = CNN(
            num_classes=model_cfg.get("num_classes", 10),
            dropout=model_cfg.get("dropout", 0.3),
            use_batchnorm=model_cfg.get("use_batchnorm", False),
        )
    elif model_name == "cnn_residual":
        model = ResidualCNN(
            num_classes=model_cfg.get("num_classes", 10),
            dropout=model_cfg.get("dropout", 0.3),
            use_batchnorm=model_cfg.get("use_batchnorm", False),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    init_method = model_cfg.get("init_method")
    if init_method is not None:
        initialize_weights(model, init_method)

    return model
