import torch
import torch.nn as nn
import sys
sys.path.insert(0,'../')
from .utils import convolution


class offsets_head(nn.Module):
    def __init__(self):
        super(offsets_head, self).__init__()
        stacks = 3
        self.conv_kernel = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(stacks)
        ])

        self.conv_search = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(stacks)
        ])
        self.offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])

    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )