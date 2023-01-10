import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['_init_weight', '_make_divisible']


def _init_weight(module: nn.Module):
    for child in module.children():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight, a=1)
            if child.bias is not None:
                nn.init.constant_(child.bias, 0)
        elif isinstance(child, nn.Sequential):
            _init_weight(child)


def _make_divisible(x, divisor=8, min_x=None):
    if min_x is None:
        min_x = divisor
    new_x = max(min_x, int(x + divisor / 2) // divisor * divisor)
    if new_x < 0.9 * x:
        new_x += divisor
    return new_x
