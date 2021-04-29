# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
from .atts.utils import _tranpose_and_gather_feat
from pysot.core.config import cfg
import torch.nn as nn

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

def _focal_loss(preds, gt): # gt:[5,1,16,16]
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4) # 1271

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def atts_loss(outs, targets):

    atts = outs # [32,1,25,25]*3
    gt_atts = targets # [32,1,25,25]
    atts = [_sigmoid(att) for att in atts]
    #atts = [[att[ind] for att in atts] for ind in range(len(gt_atts))]

    att_loss = 0
    att_loss += _focal_loss(atts, gt_atts) / max(len(atts), 1)
    return att_loss.unsqueeze(0)

def _focal_loss_mask(preds, gt, mask):  # tl_heat的loss
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_mask = mask[pos_inds]
    neg_mask = mask[neg_inds]

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * pos_mask
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights * neg_mask

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def corner_loss(outs, targets):
    tl_heats = outs[0]
    br_heats = outs[1]
    #tl_tags = outs[2]
    #br_tags = outs[3]
    tl_offs = outs[2]
    br_offs = outs[3]

    gt_tl_heat = targets[0]
    gt_br_heat = targets[1]
    gt_tl_valid = targets[2]
    gt_br_valid = targets[3]
    gt_mask = targets[4]
    gt_tl_off = targets[5]
    gt_tl_ind = targets[6]
    gt_br_off = targets[7]
    gt_br_ind = targets[8]

    focal_loss = 0

    tl_heats = [_sigmoid(t) for t in tl_heats]
    br_heats = [_sigmoid(b) for b in br_heats]
    focal_loss += _focal_loss_mask(tl_heats, gt_tl_heat, gt_tl_valid) # focal_loss = _focal_loss_mask
    focal_loss += _focal_loss_mask(br_heats, gt_br_heat, gt_br_valid)

    '''
    # tag loss
    pull_loss = 0
    push_loss = 0
    tl_tags = [_tranpose_and_gather_feat(tl_tag, gt_tl_ind) for tl_tag in tl_tags]
    br_tags = [_tranpose_and_gather_feat(br_tag, gt_br_ind) for br_tag in br_tags]
    for tl_tag, br_tag in zip(tl_tags, br_tags):
        pull, push = _ae_loss(tl_tag, br_tag, gt_mask)
        pull_loss += pull
        push_loss += push
    pull_loss = cfg.corners.pull_weight * pull_loss
    push_loss = cfg.corners.push_weight * push_loss
    '''

    off_loss = 0
    tl_offs = [_tranpose_and_gather_feat(tl_off, gt_tl_ind) for tl_off in tl_offs]  # [28,2,25,25]*3-->[28,1,2]
    br_offs = [_tranpose_and_gather_feat(br_off, gt_br_ind) for br_off in br_offs]
    for tl_off, br_off in zip(tl_offs, br_offs):
        off_loss += _off_loss(tl_off, gt_tl_off, gt_mask)
        off_loss += _off_loss(br_off, gt_br_off, gt_mask)
    off_loss = cfg.corners.off_weight * off_loss

    loss = (focal_loss + off_loss) / max(len(tl_heats), 1)
    return loss.unsqueeze(0), focal_loss.unsqueeze(0)/ max(len(tl_heats), 1), off_loss.unsqueeze(0)/ max(len(tl_heats), 1)

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _off_loss(off, gt_off, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_off)  # mask:[28,1,2]

    off = off[mask]  # off:[28,1,2] -->[56]
    gt_off = gt_off[mask] # gt_off:[28,1,2]

    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")
    off_loss = off_loss / (num + 1e-4)
    return off_loss