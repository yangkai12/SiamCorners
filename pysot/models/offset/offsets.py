import torch
import torch.nn as nn

from ..atts.utils import convolution
from pysot.core.xcorr import xcorr_depthwise


class offs_head(nn.Module):
    def __init__(self):
        super(offs_head, self).__init__()
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

    def forward(self, kernel, search):
        kernel = [self.conv_kernel_(kernel_) for self.conv_kernel_, kernel_ in zip(self.conv_kernel, kernel)]
        search = [self.conv_search_(search_) for self.conv_search_, search_ in zip(self.conv_search, search)]
        feature = [xcorr_depthwise(search_, kernel_) for search_, kernel_ in zip(search, kernel)] # depth_wise卷积 feature:[32,256,25,25] 不同深度特征结合
        offs = [off(f) for off, f in zip(self.offs, feature)] # offs:[28,2,25,25]*3
        return offs