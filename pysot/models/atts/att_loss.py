import torch
import torch.nn as nn


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

class atts_loss(nn.Module):
    def __init__(self):
        super(atts_loss, self).__init__()

    def forward(self, outs, targets):
        atts = outs # [32,1,25,25]*3
        gt_atts = targets # [32,1,25,25]
        atts = [_sigmoid(att) for att in atts]
        #atts = [[att[ind] for att in atts] for ind in range(len(gt_atts))]

        att_loss = 0
        att_loss += _focal_loss(atts, gt_atts) / max(len(atts), 1)
        return att_loss.unsqueeze(0)