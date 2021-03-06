# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from matplotlib import pyplot as plt
from .external.nms import soft_nms


class SiamCORNERTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamCORNERTracker, self).__init__()
        #self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
        #    cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        #self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        #hanning = np.hanning(self.score_size)
        window = np.hanning(cfg.corners.hanning_score_size)
        #window = np.outer(hanning, hanning)
        self.window_ = np.tile(window.flatten(), cfg.corners.hanning_num)
        #self.anchors = self.generate_anchor(self.score_size)
        self.window = self.window_[: cfg.corners.hanning_score_len]
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1) # delta:[4,3125]
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0] # x
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1] # y
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2] # w
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3] # h
        return delta

    def _convert_score_att(self, atts, attentions):
        att_score = []
        att_loc = []
        for att, attention in zip(atts, attentions):
            attention_ = attention.repeat(1, 5, 1, 1) # ??????5???[1,1,25,25]-->[1,5,25,25] 5????????????5????????????anchors
            attention_ = attention_.permute(1, 2, 3, 0).contiguous().view(1, -1).permute(1, 0) # [3125,1]
            att_score.append(F.softmax(attention_, dim=0).data[:, 0].cpu().numpy()) # [3125,] ?????????????????????softmax????????????            A = []
            for i in range(len(att)): # ???????????????????????????????????????????????????
                a_0 = (att[i][0]*25 + att[i][1]).astype(int) # ?????????[3125,]????????????
                a_1 = a_0 + 25*25*1
                a_2 = a_1 + 25*25*1
                a_3 = a_2 + 25*25*1
                a_4 = a_3 + 25*25*1
                A.append(a_0)
                A.append(a_1)
                A.append(a_2)
                A.append(a_3)
                A.append(a_4)
            att_loc.append(A)

        return att_loc, att_score

    def _convert_score(self, score, att_locs, att_scores):
        W = cfg.atts.W # ?????????????????????        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0) # [1,10,25,25]-->[3125,2] ????????????????????????        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy() # [3125,] dim=1 ??????????????????softmax
        for att_loc, att_score, w in zip(att_locs, att_scores, W):
            for i in range(len(att_loc)):
                score[att_loc[i]] += w * att_score[att_loc[i]]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def rescale_dets_(self, dets_):
        input_size = cfg.corners.input_size  # [255,255]
        output_size = cfg.corners.output_size  # [25,25]

        ratios = [o / i for o, i in zip(output_size, input_size)]
        detect = []
        for dets in dets_:
            dets[..., 0:4:2] /= ratios[1]  # bbox???[255,255,3]????????????
            dets[..., 1:4:2] /= ratios[0]
            detect.append(dets)
        return detect

    def Centertocorner(self, det):
        det[:, :, 0] = det[:, :, 0] - det[:, :, 2] // 2 # x1
        det[:, :, 1] = det[:, :, 1] - det[:, :, 3] // 2  # y1
        det[:, :, 2] = det[:, :, 0] + det[:, :, 2]  # x2
        det[:, :, 3] = det[:, :, 1] + det[:, :, 3]  # y2
        return det

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        # visual bounding box
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,  # center_x, center_y
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])  # w,h

        bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]  # [x1,y1,x2,y2]
        #cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255),
        #              thickness=1)  # ???[0,0,255]
        cv2.imwrite('Before_crop.jpg', img)  # image:[255,255,3]

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size) # ??????????????????w+p
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z)) # the scale of proposal

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1)) # ?????????
        # get crop
        im_t, im_l, im_b, im_r, bbox = self.get_subwindow(img, self.center_pos, # self.center_pos:ground_truth???????????????                                   cfg.TRACK.EXEMPLAR_SIZE, # ??????ground_truth??????????????????127*127?????????                                    s_z, self.channel_average, bbox)#[1024,1024,3]-->z_crop:[1,3,127,127]
        # ??????
        img1 = im_t.squeeze(0)
        img_ = img1.transpose(1, 0).transpose(2, 1).cpu().numpy()
        cv2.imwrite("im_t.jpg", img_)  # save picture

        # ??????
        img1 = im_l.squeeze(0)
        img_ = img1.transpose(1, 0).transpose(2, 1).cpu().numpy()
        cv2.imwrite("im_l.jpg", img_)  # save picture

        # ??????
        img1 = im_b.squeeze(0)
        img_ = img1.transpose(1, 0).transpose(2, 1).cpu().numpy()
        cv2.imwrite("im_b.jpg", img_)  # save picture

        # ??????
        img1 = im_r.squeeze(0)
        img_ = img1.transpose(1, 0).transpose(2, 1).cpu().numpy()
        cv2.imwrite("im_r.jpg", img_)  # save picture

        # visual bounding box
        #bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        #cv2.rectangle(img_, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255),
        #              thickness=1)  # ???[0,0,255]
        #cv2.imwrite('After_crop.jpg', img_)  # image:[255,255,3]
        self.model.template(im_t, im_l, im_b, im_r)


    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size) # self.size[0]:bbox??? w+p
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size) # self.size[1]:bbox??? h+p
        s_z = np.sqrt(w_z * h_z) # ?????????????????????4
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,  # x_crop:[1,3,255,255] ???????????????????????????                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average, bbox=None)
        # ??????
        img1 = x_crop.squeeze(0)
        img_ = img1.transpose(1, 0).transpose(2, 1).cpu().numpy()
        cv2.imwrite("search.jpg", img_)  # save picture

        outputs = self.model.track(x_crop) # cls:[1,10,25,25] loc:[1,20,25,25]
        dets_ = outputs['bbox']
        dets_1 = self.rescale_dets_(dets_)  # [25,25]-->[255,255] ?????????bbox???????????? [x1,y1,x2,y2]

        det = []
        for dets_2 in dets_1:
            dets_2 = dets_2.cpu()
            dets_2 = dets_2.detach().numpy()
            dets_2 = np.squeeze(dets_2)
            det.append(dets_2)

        detectect = np.concatenate((det[0], det[1]))
        det = np.concatenate((detectect, det[2])) # ???[255,255]???????????????[x1,y1,x2,y2]
        det[:, :4] = det[:, :4] / scale_z  # ???padding??????????????????

        det[:, 2], det[:, 3] = det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]  # ?????????padding?????????        det[:, 0], det[:, 1] = det[:, 0] + det[:, 2] // 2, det[:, 1] + det[:, 3] // 2  # ???????????????x,???????????????y
        s_x = round((s_x + 1) // 2)  # padding?????????        det[:, 0], det[:, 1] = det[:, 0] - s_x, det[:, 1] - s_x  # ???????????????x,y????????????
        det[:, 0] = det[:, 0] + self.center_pos[0]  # ???????????????????????????????????????x
        det[:, 1] = det[:, 1] + self.center_pos[1]  # ????????????????????????y

        delt_x = abs(det[:, 0] - self.center_pos[0]) # ??????????????????
        delt_y = abs(det[:, 1] - self.center_pos[1])  # ??????????????????
        delt_xy = delt_x + delt_y

        delt_w = abs(det[:, 2] - self.size[0])  # ?????????        delt_h = abs(det[:, 3] - self.size[1])  # ?????????        delt_wh = delt_h + delt_w

        delt_xywh = 0.5*delt_xy + 0.5*delt_wh
        sort_dis = delt_xywh.argsort()[::-1]  # ???????????????????????????????????????
        #best_idx = np.argmin(delt_wh)
        def change(r):  # r:[3125,]
            for i in range(cfg.corners.hanning_score_len):
                if r[i] == 0:
                    r[i] = 1e-4
            return np.maximum(r, (1. / (r)))# ??????

        def sz(w, h):
            pad = (w + h) * 0.5
            if type(w) is np.ndarray:
                sqrt_value = np.zeros(cfg.corners.hanning_score_len, dtype=np.float32)
                for i in range(cfg.corners.hanning_score_len):
                    if (w[i] + pad[i]) * (h[i] + pad[i]) < 0:
                        sqrt_value[i] = 0
                    else:
                        sqrt_value[i] = (w[i] + pad[i]) * (h[i] + pad[i])
                return np.sqrt(sqrt_value)  # ?????????????????????15)
            else:
                return np.sqrt((w + pad) * (h + pad)) # ??????14



        # scale penalty   ??????s
        s_c = change(sz(det[:, 2], det[:, 3]) /  # s_c:[3125,] det?????????bbox???????????? self.size[0]???self.size[1]?????????????????????
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ?????????bbox??????????????????

        # aspect ratio penalty  ??????r
        r_c = change((self.size[0] / self.size[1]) /  # self.size:bbox???????????????????????????                     (det[:, 2] / det[:, 3]))

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  # penalty:[3125,]
        score = det[:, 4]
        pscore = penalty * score  # ???????????????????????????????????????????????????????????????
        for i in range(cfg.corners.hanning_score_len):

            pscore[sort_dis[i]] = pscore[sort_dis[i]] * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window[i] * cfg.TRACK.WINDOW_INFLUENCE  # a cosine window is added to suppress the large displacement

        best_idx = np.argmax(pscore)
        bbox = det[best_idx, :4]  # pred_bbox:[4,3125] ??????padding [x1,y1,x2,y2]
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr  # self.size[0]:bbox??????????????????
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(bbox[0], bbox[1], width,  # ????????????????????????????????????
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = det[best_idx, 4]
        return {
            'bbox': bbox,
            'best_score': best_score
        }


