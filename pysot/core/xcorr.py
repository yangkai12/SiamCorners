# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0) # batch:58 kernel:[58,256,31,31]
    channel = kernel.size(1) # channel：256
    x = x.view(1, batch*channel, x.size(2), x.size(3)) # x：[1,14838,31,31]
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3)) # kernel：[14848,1, 7, 7]
    out = F.conv2d(x, kernel, groups=batch*channel)
    #out = out.cpu()
    #out_ = out.detach().numpy()
    #plt.imshow(out_[0, 0])
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
