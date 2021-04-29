# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import math

from pysot.utils.bbox import corner2center, \
        Center, center2corner, Corner
from pysot.core.config import cfg
from .utils import draw_gaussian, gaussian_radius, normalize_, color_jittering_, lighting_, crop_image
import matplotlib.pyplot as plt
from PIL import Image

class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)
        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox

    def _flip_aug(self, image, bbox):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def create_attention_mask(self, atts, ratios, detections):  # ratios = 255/25=0.2
        x = (detections[0] + detections[2]) / 2  # 中心点
        y = (detections[1] + detections[3]) / 2
        x_int = (x / ratios).astype(np.int32)
        y_int = (y / ratios).astype(np.int32)  # in:[255,255]的bbox在[25, 25]的attention map上的缩放比例
        for att in atts:
            att[0, y_int, x_int] = 1
        x_float = x / ratios
        y_float = y / ratios
        return att, x_int, y_int, x_float, y_float

    def __call__(self, image, bbox, size, data, gray=False):
        shape = image.shape
        cv2.imwrite('511.jpg', image) # image：[511,511,3]

        if data == 'template':
            image1 = np.zeros((127, 127, 3))
            for i in range(127):
                for j in range(127):
                    for k in range(3):
                        if k == 0:
                            image1[i, j, k] = 87
                        elif k == 1:
                            image1[i, j, k] = 135
                        elif k == 2:
                            image1[i, j, k] = 123


        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)
        #cv2.imwrite('127_255.jpg', image)  # image：[127,127,3] 或 [255,255,3]
        crop_bbox = center2corner(Center(shape[0] // 2, shape[1] // 2,
                                         size - 1, size - 1))
        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)

        if data == 'template':
            # visual bounding box
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255),
                          thickness=1)  # 红[0,0,255]
            cv2.imwrite('127_bbox.jpg', image)  # image:[255,255,3]

            image_l = image1
            image_t = image1.copy()
            image_b = image1.copy()
            image_r = image1.copy()
            image_l[int(bbox[1]):(int(bbox[3])), int(bbox[0]):(int(bbox[0]+cfg.corners.crop_size)), :] =\
                image[int(bbox[1]):(int(bbox[3])), int(bbox[0]):(int(bbox[0]+cfg.corners.crop_size)), :]
            cv2.imwrite('crop_l.jpg', image_l)  # image:[255,255,3]

            #cv2.imwrite('127_bbox——2.jpg', image)  # image:[255,255,3]
            #cv2.imwrite('127_bbox--3.jpg', image_t)  # image:[255,255,3]
            image_t[int(bbox[1]):(int(bbox[1]+cfg.corners.crop_size)), int(bbox[0]):(int(bbox[2])), :] = \
                image[int(bbox[1]):(int(bbox[1]+cfg.corners.crop_size)), int(bbox[0]):(int(bbox[2])), :]
            cv2.imwrite('crop_t.jpg', image_t)  # image:[255,255,3]

            image_b[(int(bbox[3] - cfg.corners.crop_size)):int(bbox[3]), int(bbox[0]):(int(bbox[2])), :] = \
                image[(int(bbox[3] - cfg.corners.crop_size)):int(bbox[3]), int(bbox[0]):(int(bbox[2])), :]
            cv2.imwrite('crop_b.jpg', image_b)  # image:[255,255,3]

            image_r[int(bbox[1]):(int(bbox[3])), (int(bbox[2]-cfg.corners.crop_size)):int(bbox[2]), :] = \
                image[int(bbox[1]):(int(bbox[3])), (int(bbox[2]-cfg.corners.crop_size)):int(bbox[2]), :]
            cv2.imwrite('crop_r.jpg', image_r)  # image:[255,255,3]


        if data == 'search':
            attentions = [np.zeros((1, cfg.atts.att_size, cfg.atts.att_size), dtype=np.float32)]  # 25 为attention map大小
            # tl_heats_map
            tl_heats = np.zeros((1, cfg.corners.cor_size, cfg.corners.cor_size),
                                dtype=np.float32)  # [1,25,25]
            br_heats = np.zeros((1, cfg.corners.cor_size, cfg.corners.cor_size), dtype=np.float32)
            # tl_valids
            tl_regrs = np.zeros((cfg.corners.offs_max_objects, 2), dtype=np.float32)
            br_regrs = np.zeros((cfg.corners.offs_max_objects, 2), dtype=np.float32)
            tl_tags = np.zeros((cfg.corners.offs_max_objects), dtype=np.int64)
            br_tags = np.zeros((cfg.corners.offs_max_objects), dtype=np.int64)
            tl_valids = np.zeros((1, cfg.corners.cor_size, cfg.corners.cor_size),
                                 dtype=np.float32)  # [1,25,25]
            br_valids = np.zeros((1, cfg.corners.cor_size, cfg.corners.cor_size), dtype=np.float32)
            tag_masks = np.ones((cfg.corners.offs_max_objects), dtype=np.uint8)
            tag_lens = 0

            #atts_map, x_int, y_int, x_float, y_float = self.create_attention_mask(attentions, cfg.TRAIN.ratios, bbox) # image:[255,255,3] x_int,y_int为目标中心点坐标
            atts_map =[]

            xtl, ytl = bbox[0], bbox[1]  # 图大小为255的坐标
            xbr, ybr = bbox[2], bbox[3]

            det_height = int(ybr) - int(ytl)
            det_width = int(xbr) - int(xtl)
            det_max = max(det_height, det_width)

            min_scale = 16
            valid = det_max >= min_scale  # min_scale:16

            fxtl = (xtl * cfg.corners.Ratios)  # width_ratio:由255-->25的缩放比例
            fytl = (ytl * cfg.corners.Ratios)
            fxbr = (xbr * cfg.corners.Ratios)
            fybr = (ybr * cfg.corners.Ratios)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # visual bounding box
            #cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=1)  # 红[0,0,255]
            #cv2.imwrite('255.jpg', image) # image:[255,255,3]


            width = math.ceil(width * cfg.corners.Ratios)
            height = math.ceil(height * cfg.corners.Ratios)

            if cfg.corners.gaussian_rad == -1:
                radius = gaussian_radius((height, width), cfg.corners.gaussian_iou)
                radius = max(0, int(radius))
            else:
                radius = cfg.corners.gaussian_rad

            if valid:
                draw_gaussian(tl_heats[0], [xtl, ytl], radius)
                draw_gaussian(br_heats[0], [xbr, ybr], radius)
                tl_regrs[0, :] = [fxtl - xtl, fytl - ytl]  # tl_regrs:[5,128,2]
                br_regrs[0, :] = [fxbr - xbr, fybr - ybr]
                tl_tags[0] = max(0, min(ytl * cfg.corners.cor_size + xtl, cfg.corners.cor_size*cfg.corners.cor_size-1)) # 坐标索引 ytl为取整后
                br_tags[0] = max(0, min(ybr * cfg.corners.cor_size + xbr, cfg.corners.cor_size*cfg.corners.cor_size-1))
            else:
                draw_gaussian(tl_valids[b_ind, category], [xtl, ytl], radius)  # 得到上左masked_heatmap
                draw_gaussian(br_valids[b_ind, category], [xbr, ybr], radius)

            tl_valids = (tl_valids == 0).astype(np.float32)
            br_valids = (br_valids == 0).astype(np.float32)

            #tag_masks[:1] = 1


        else:
            atts_map, tl_heats, br_heats, tl_valids, br_valids, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags = [], [],\
            [], [], [], [], [], [], [], []

        '''
        if x_int:
            tag_masks = np.ones((cfg.offs.max_objects), dtype=np.uint8)
            tl_regrs  = np.zeros((cfg.offs.max_objects, 2), dtype=np.float32)  # max_objects:1
            tl_regrs[0, :] = [x_float - x_int, y_float - y_int]  # tl_regrs:[5,128,2]
            tl_tags = np.zeros((cfg.offs.max_objects), dtype=np.int64)
            tl_tags[0] = y_int * cfg.offs.off_size + x_int  # 坐标索引 ytl为取整后
        else:
            tl_heats, br_heats, tl_valids, br_valids, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags = [], [], [], [], [], [], [], [], []
        '''
        if data == 'template':
            return image_t, image_l, image_b, image_r, bbox, atts_map, tl_heats, br_heats, tl_valids, br_valids, tag_masks, tl_regrs, br_regrs, tl_tags,\
                br_tags
        else:
            return image, bbox, atts_map, tl_heats, br_heats, tl_valids, br_valids, tag_masks, tl_regrs, br_regrs, tl_tags, \
                   br_tags
