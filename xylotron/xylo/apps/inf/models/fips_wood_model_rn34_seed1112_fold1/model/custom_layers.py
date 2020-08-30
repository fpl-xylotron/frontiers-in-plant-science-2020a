import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=None):
        super().__init__()
        output_size = output_size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
