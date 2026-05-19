import torch.nn as nn


def initialize_weights(model, method="he"):
    """
    对模型参数进行初始化
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                pass

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
