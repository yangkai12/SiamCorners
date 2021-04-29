# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
import time

from pysot.core.config import cfg


class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans, bbox):
        
        # convert bbox to [x1,y1,x2,y2]
        #bbox_w, bbox_h = bbox[2], bbox[3]
        #bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        if isinstance(pos, float): # pos为bbox的中心位�?            
            pos = [pos, pos]
        sz = original_sz  # w+p 抠图大小
        im_sz = im.shape  # 1024
        c = (original_sz + 1) / 2 # padding之后的宽或高的一�?        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)  # bbox的中�?抠图宽的一�?看bbox出了边界�?        
        context_xmax = context_xmin + sz - 1  # padding之后的起�?抠图�?        
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin)) # contest_xmin不能为负�?left_pad表示左侧填充多少而不为负�?        
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape # 原图
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k) # c为宽
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if bbox is not None:  # 计算bbox在padding之后的位�?                
                bbox[0], bbox[2] = bbox[0] + left_pad, bbox[2] + left_pad
                bbox[1], bbox[3] = bbox[1] + top_pad, bbox[3] + top_pad

            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1), # paddin之后，取出对应的bbox模块
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]  # padding之后，取出对应的bbox模块

        if bbox is not None:
            # 画图
            bbox[0], bbox[2] = bbox[0] - context_xmin, bbox[2] - context_xmin # bbox:padding之后的位�?
            bbox[1], bbox[3] = bbox[1] - context_ymin, bbox[3] - context_ymin
            cv2.rectangle(im_patch, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255),
                          thickness=1)  # 红[0,0,255]
            #cv2.imwrite("croped.jpg", im_patch)  # save picture
            w_, h_, z = im_patch.shape
            size1 = (w_, h_, z)

            im_tm_t = np.zeros(size1, np.uint8)
            im_tm_t[:, :, :] = avg_chans
            crop_size_t = cfg.corners.crop_size * (bbox[3]-bbox[1])
            im_tm_t[int(bbox[1]):(int(bbox[1]+crop_size_t)), int(bbox[0]):(int(bbox[2])), :] =\
                im_patch[int(bbox[1]):(int(bbox[1]+crop_size_t)), int(bbox[0]):(int(bbox[2])), :]
            #cv2.imwrite("crop_t.jpg", im_tm_t)  # save picture

            im_tm_l = np.zeros(size1, np.uint8)
            im_tm_l[:, :, :] = avg_chans
            crop_size_l = cfg.corners.crop_size * (bbox[2]-bbox[0])
            im_tm_l[int(bbox[1]):(int(bbox[3])), int(bbox[0]):(int(bbox[0]+crop_size_l)), :] = \
                im_patch[int(bbox[1]):(int(bbox[3])), int(bbox[0]):(int(bbox[0]+crop_size_l)), :]
            #cv2.imwrite("crop_l.jpg", im_tm_l)  # save picture

            im_tm_b = np.zeros(size1, np.uint8)
            im_tm_b[:, :, :] = avg_chans
            crop_size_b = cfg.corners.crop_size * (bbox[3]-bbox[1])
            im_tm_b[(int(bbox[3] - crop_size_b)):int(bbox[3]), int(bbox[0]):(int(bbox[2])), :] = \
                im_patch[(int(bbox[3] - crop_size_b)):int(bbox[3]), int(bbox[0]):(int(bbox[2])), :]
            #cv2.imwrite("crop_b.jpg", im_tm_b)  # save picture

            im_tm_r = np.zeros(size1, np.uint8)
            im_tm_r[:, :, :] = avg_chans
            crop_size_r = cfg.corners.crop_size * (bbox[2]-bbox[0])
            im_tm_r[int(bbox[1]):(int(bbox[3])), (int(bbox[2]-crop_size_r)):int(bbox[2]), :] = \
                im_patch[int(bbox[1]):(int(bbox[3])), (int(bbox[2]-crop_size_r)):int(bbox[2]), :]
            #cv2.imwrite("crop_r.jpg", im_tm_r)  # save picture
            #time.sleep(150)

            if not np.array_equal(model_sz, original_sz):
                im_t = cv2.resize(im_tm_t, (model_sz, model_sz))  # 扣出来的�?->[127,127,3]
                im_l = cv2.resize(im_tm_l, (model_sz, model_sz))  # 扣出来的�?->[127,127,3]
                im_b = cv2.resize(im_tm_b, (model_sz, model_sz))  # 扣出来的�?->[127,127,3]
                im_r = cv2.resize(im_tm_r, (model_sz, model_sz))  # 扣出来的�?->[127,127,3]
            else:
                im_t = im_tm_t
                im_l = im_tm_l
                im_b = im_tm_b
                im_r = im_tm_r
                #im_patch = cv2.resize(im_patch, (model_sz, model_sz)) # 扣出来的�?->[127,127,3]
            #im_patch = im_patch.transpose(2, 0, 1)
            #im_patch = im_patch[np.newaxis, :, :, :]
            #im_patch = im_patch.astype(np.float32)
            #im_patch = torch.from_numpy(im_patch)
            #if cfg.CUDA:
            #    im_patch = im_patch.cuda()

            im_t = im_t.transpose(2, 0, 1)
            im_t = im_t[np.newaxis, :, :, :]
            im_t = im_t.astype(np.float32)
            im_t = torch.from_numpy(im_t)
            if cfg.CUDA:
                im_t = im_t.cuda()

            im_l = im_l.transpose(2, 0, 1)
            im_l = im_l[np.newaxis, :, :, :]
            im_l = im_l.astype(np.float32)
            im_l = torch.from_numpy(im_l)
            if cfg.CUDA:
                im_l = im_l.cuda()

            im_b = im_b.transpose(2, 0, 1)
            im_b = im_b[np.newaxis, :, :, :]
            im_b = im_b.astype(np.float32)
            im_b = torch.from_numpy(im_b)
            if cfg.CUDA:
                im_b = im_b.cuda()

            im_r = im_r.transpose(2, 0, 1)
            im_r = im_r[np.newaxis, :, :, :]
            im_r = im_r.astype(np.float32)
            im_r = torch.from_numpy(im_r)
            if cfg.CUDA:
                im_r = im_r.cuda()
            return im_t, im_l, im_b, im_r, bbox

        if bbox is None:
            if not np.array_equal(model_sz, original_sz):
                im_patch = cv2.resize(im_patch, (model_sz, model_sz))
            im_patch = im_patch.transpose(2, 0, 1)
            im_patch = im_patch[np.newaxis, :, :, :]
            im_patch = im_patch.astype(np.float32)
            im_patch = torch.from_numpy(im_patch)
            if cfg.CUDA:
                im_patch = im_patch.cuda()
            return im_patch

