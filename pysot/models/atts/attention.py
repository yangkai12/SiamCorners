import torch
import torch.nn as nn
from .utils import convolution
from pysot.core.xcorr import xcorr_depthwise

class atts_head(nn.Module):
    def __init__(self):
        super(atts_head, self).__init__()
        stacks = 3
        self.conv_kernel = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )   for _ in range(stacks)
        ])

        self.conv_search = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )   for _ in range(stacks)
        ])

        self.att_mods = nn.ModuleList([
                        nn.Sequential(
                            convolution(3, 256, 256, with_bn=False),
                            nn.Conv2d(256, 1, (1, 1))
                        )  for _ in range(stacks)
                ])

        for att in self.att_mods:
                torch.nn.init.constant_(att[-1].bias, -2.19) # 初始化为常数

    def forward(self, kernel, search):
        kernel = [self.conv_kernel_(kernel_) for self.conv_kernel_, kernel_ in zip(self.conv_kernel, kernel)]
        search = [self.conv_search_(search_) for self.conv_search_, search_ in zip(self.conv_search, search)]
        feature = [xcorr_depthwise(search_, kernel_) for search_, kernel_ in zip(search, kernel)] # depth_wise卷积 feature:[32,256,25,25] 不同深度特征结合
        atts = [att(f) for att, f in zip(self.att_mods, feature)] # atts:[32,1,25,25]*3
        return atts
