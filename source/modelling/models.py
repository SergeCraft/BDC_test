from timm.models import byobnet, efficientnet, resnet
from torch import nn


def get_model(n_classes: int, name: str, pretrained: bool, **kwargs) -> nn.Module:
    names = {'resnet18', 'efficientnetv2_s', 'gernet_s'}
    if name not in names:
        raise ValueError(f'Model name should be one of these: {names}')

    if name == 'resnet18':
        return resnet.resnet18(pretrained, num_classes=n_classes, **kwargs)
    elif name == 'efficientnetv2_s':
        return efficientnet.efficientnetv2_s(pretrained, num_classes=n_classes, **kwargs)
    elif name == 'gernet_s':
        return byobnet.gernet_s(pretrained, num_classes=n_classes, **kwargs)
