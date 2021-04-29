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
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        #hanning = np.hanning(self.score_size)
        window = np.hanning(cfg.corners.hanning_score_size)
        #window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), cfg.corners.hanning_num)
        #self.anchors = self.generate_anchor(self.score_size)
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
            attention_ = attention.repeat(1, 5, 1, 1) # 复制5份[1,1,25,25]-->[1,5,25,25] 5份是因为5种不同的anchors
            attention_ = attention_.permute(1, 2, 3, 0).contiguous().view(1, -1).permute(1, 0) # [3125,1]
            att_score.append(F.softmax(attention_, dim=0).data[:, 0].cpu().numpy()) # [3125,] 注意力特征经过softmax激活函数
            A = []
            for i in range(len(att)): # 注意标注的位置在复制的5份对应的位置
                a_0 = (att[i][0]*25 + att[i][1]).astype(int) # 转化为[3125,]后的位置
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
        W = cfg.atts.W # 注意力机制权值
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0) # [1,10,25,25]-->[3125,2] 后5个为正样本得分
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy() # [3125,] dim=1 对每一行进行softmax
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
            dets[..., 0:4:2] /= ratios[1]  # bbox在[255,255,3]中的坐标
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
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2, # center_x, center_y
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]]) # w,h

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, # self.center_pos:ground_truth中心点位置
                                    cfg.TRACK.EXEMPLAR_SIZE, # 按照ground_truth中心点裁剪为127*127的图片
                                    s_z, self.channel_average) # image:[1024,1024,3]-->z_crop:[1,3,127,127]
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size) # self.size[0]:bbox宽  w+p
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size) # self.size[1]:bbox高  h+p
        s_z = np.sqrt(w_z * h_z) # 对应论文中公式14
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,  # x_crop:[1,3,255,255] 按照中心点进行裁剪
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        '''
        # 画图
        img1 = x_crop.squeeze(0)
        img_1 = img1.transpose(1, 0).transpose(2, 1).cpu().numpy()
        a = cv2.imwrite("4.jpg", img_1)  # save picture
        plt.axis('off')
        plt.imshow(img_1[:, :, 1])
        plt.show()
        '''

        outputs = self.model.track(x_crop) # cls:[1,10,25,25] loc:[1,20,25,25]
        dets_ = outputs['bbox']
        dets_1 = self.rescale_dets_(dets_)  # [25,25]-->[255,255] 对应的bbox缩放比例 [x1,y1,x2,y2]

        det = []
        for dets_2 in dets_1:
            dets_2 = dets_2.cpu()
            dets_2 = dets_2.detach().numpy()
            dets_2 = np.squeeze(dets_2)
            det.append(dets_2)

        detectect = np.concatenate((det[0], det[1]))
        det = np.concatenate((detectect, det[2])) # 在[255,255]中的位置，[x1,y1,x2,y2]


        def change(r): # r:[3125,]
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad)) # 对应论文中公式(15)

        # scale penalty
        s_c = change(sz(det[:, 2], det[:, 3]) / # s_c:[3125,]
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))# 上一帧bbox在填充之后的

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) / # self.size:bbox代表上一帧宽，高比
                     (det[:, 2]/det[:, 3]))

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  # penalty:[3125,]
        score = det[:, 5]
        #pscore = penalty * score  # 对相邻两帧之间过大的形状和尺度变化进行抑制

        # window penalty
        #pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
        #         self.window * cfg.TRACK.WINDOW_INFLUENCE  # a cosine window is added to suppress the large displacement
        best_idx = np.argmax(score)
        bbox = det[best_idx, :4] / scale_z  # pred_bbox:[4,3125] 对应padding [x1,y1,x2,y2]
        #lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        lr = 0.13104

        bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]  # 对应于padding宽，高
        bbox[0], bbox[1] = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2  # 中心点坐标x,中心点坐标y
        s_x = round((s_x + 1) // 2)
        bbox[0], bbox[1] = bbox[0] - s_x, bbox[1] - s_x  # 中心点坐标x,y的偏移量
        bbox[0] = bbox[0] + self.center_pos[0]  # 对应于原图中心点在原图坐标x
        bbox[1] = bbox[1] + self.center_pos[1]  # 中心点在原图坐标y

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr  # self.size[0]:bbox代表上一帧宽
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(bbox[0], bbox[1], width,  # 目标框框不能超过图片边界
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score
        }




'''

        detection = []  # 对应于padding后的坐标
        for det1 in dets_1:
            det1[..., 0:4] /= scale_z
            detection.append(det1) # 对应于padding后的坐标 x1,y1,x2,y2

        det = []

        for det_ in detection: # 在原图位置
            det_[:, :, 2], det_[:, :, 3] = det_[:, :, 2]-det_[:, :, 0], det_[:, :, 3]-det_[:, :, 1] # 宽，高
            det_[:, :, 0], det_[:, :, 1] = det_[:, :, 0] + det_[:, :, 2] // 2, det_[:, :, 1] + det_[:, :, 3] // 2# 中心点坐标x,中心点坐标y
            s_x = round((s_x+1)//2)
            det_[:, :, 0], det_[:, :, 1] = det_[:, :, 0] - s_x, det_[:, :, 1] - s_x  # 中心点坐标x,y的偏移量
            det_[:, :, 0] = det_[:, :, 0] + self.center_pos[0] # 中心点在原图坐标x
            det_[:, :, 1] = det_[:, :, 1] + self.center_pos[1] # 中心点在原图坐标y
            detection = self.Centertocorner(det_) # x1,y1,x2,y2

            det.append(detection)

        detectection = []
        for detect in det: # Soft nms
            detect = detect.cpu()
            detect = detect.detach().numpy()
            detect = np.squeeze(detect)
            keep_inds = soft_nms(detect, Nt=cfg.corners.nms_threshold, method=cfg.corners.nms_algorithm,
                                 sigma=0.7)  # remove redundant ones by applying Soft-NMS
            det_ = detect[keep_inds, 0:5]
            detectection.append(det_)

        detectect = np.concatenate((detectection[0], detectection[1]))
        det = np.concatenate((detectect, detectection[2]))







        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]



        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, # 目标框框不能超过图片边界
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
'''