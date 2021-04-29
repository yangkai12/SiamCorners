import torch
import torch.nn as nn
import time 
import cv2
from ..atts.utils import corner_pool, convolution
from pysot.core.xcorr import xcorr_depthwise
from .py_utils import TopPool, BottomPool, LeftPool, RightPool
from matplotlib import pyplot as plt

class corners_head(nn.Module):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def __init__(self):
        super(corners_head, self).__init__()
        stacks = 3

        self.conv_kernel_l = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )   for _ in range(stacks)
        ])

        self.conv_kernel_t = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(stacks)
        ])

        self.conv_kernel_b = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(stacks)
        ])

        self.conv_kernel_r = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in range(stacks)
        ])


        self.tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])
        self.br_modules = nn.ModuleList([corner_pool(256, BottomPool, RightPool) for _ in range(stacks)])

        self.tl_heats = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        self.br_heats = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        for tl_heat, br_heat in zip(self.tl_heats, self.br_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)

        #self.tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        #self.br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        self.tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        self.br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])

    def forward(self, kernel_t, kernel_l,  kernel_b, kernel_r, search):
        kernel_t_t = [self.conv_kernel_t_(kernel_t_) for self.conv_kernel_t_, kernel_t_ in zip(self.conv_kernel_t, kernel_t)]
        kernel_l_l = [self.conv_kernel_l_(kernel_l_) for self.conv_kernel_l_, kernel_l_ in zip(self.conv_kernel_l, kernel_l)]
        kernel_b_b = [self.conv_kernel_b_(kernel_b_) for self.conv_kernel_b_, kernel_b_ in zip(self.conv_kernel_b, kernel_b)]
        kernel_r_r = [self.conv_kernel_r_(kernel_r_) for self.conv_kernel_r_, kernel_r_ in zip(self.conv_kernel_r, kernel_r)]
        feature_t = [xcorr_depthwise(search_, kernel_t) for search_, kernel_t in zip(search, kernel_t_t)] # depth_wise卷积 feature:[32,256,25,25] 不同深度特征结合
        feature_l = [xcorr_depthwise(search_, kernel_l) for search_, kernel_l in zip(search, kernel_l_l)] # depth_wise卷积 feature:[32,256,25,25] 不同深度特征结合
        feature_b = [xcorr_depthwise(search_, kernel_b) for search_, kernel_b in zip(search, kernel_b_b)] # depth_wise卷积 feature:[32,256,25,25] 不同深度特征结合
        feature_r = [xcorr_depthwise(search_, kernel_r) for search_, kernel_r in zip(search, kernel_r_r)] # depth_wise卷积 feature:[32,256,25,25] 不同深度特征结合
        tl_modules = [tl_mod_(f_t, f_l) for tl_mod_, f_t, f_l in zip(self.tl_modules, feature_t, feature_l)]  # [28,256,25,25]-->[28,256,25,25]
        br_modules = [br_mod_(f_b, f_r) for br_mod_, f_b, f_r in zip(self.br_modules, feature_b, feature_r)]
        tl_heats = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]  # [28,256,25,25]--> [28,1,25,25]*3
        br_heats = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        #tl_tags = [tl_tag_(tl_mod) for tl_tag_, tl_mod in zip(self.tl_tags, tl_modules)]  # [5,1,64,64]*3
        #br_tags = [br_tag_(br_mod) for br_tag_, br_mod in zip(self.br_tags, br_modules)]
        tl_offs = [tl_off_(tl_mod) for tl_off_, tl_mod in zip(self.tl_offs, tl_modules)]  # [28,2,25,25]
        br_offs = [br_off_(br_mod) for br_off_, br_mod in zip(self.br_offs, br_modules)]  # [28,2,25,25]
        
        #for feature_ in  feature_t:
        #    feature_ = feature_.squeeze()
        #    feature_ = feature_.cpu()
        #    feature_ = feature_.detach().numpy()
        #    feature_ = cv2.normalize(feature_, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        #    k = 182
        #    print(feature_[k, :, :])
            #feature_[k,10,12] =1
            #feature_[k,9,12] =1
        #    for i in range(0,25):
        #      for j in range(0,25):
        #        if feature_[k,i,j] <164:
        #          feature_[k,i,j] =1
            
                  
              
           # print(feature_[k, :, :])
            
           # plt.axis('off')
           # plt.imshow(feature_[k, :, :])
           # plt.show()
           # print('yangkai')
            #cv2.imwrite("feature_.jpg", feature_)  # save picture
            #time.sleep(1500)
            
        
        return tl_heats, br_heats, tl_offs, br_offs
