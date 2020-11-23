# R-MAC Layer implementation for Pytorch
#
# Authors: Zhijian Qiao
#
# Based on the original implementation in MATLAB and https://github.com/v-v/RMAC-TensorFlow-2.git

import torch.nn.functional as F
import numpy as np
import torch
from torch.nn import AdaptiveMaxPool2d,AdaptiveAvgPool2d


class RMAC:
    def __init__(self, shape, levels=3, power=None, overlap=0.4, norm_fm=False, sum_fm=True, verbose=False):
        # shape [B,D,W,H]
        # levels - number of levels / scales at which to to generate pooling regions (default = 3)
        # power - power exponent to apply (not used by default)
        # overlap - overlap percentage between regions (default = 40%)
        # norm_fm - normalize feature maps (default = False)
        # sum_fm - sum feature maps (default = False)
        # verbose - verbose output - shows details about the regions used (default = False)

        self.shape = shape
        self.sum_fm = sum_fm
        self.norm = norm_fm
        self.power = power

        # ported from Giorgios' Matlab code
        steps = np.asarray([2, 3, 4, 5, 6, 7])
        B, D, H, W = shape
        w = min([W, H])
        w2 = w // 2 - 1
        b = np.asarray((max(H, W) - w)) / (steps - 1);
        idx = np.argmin(np.abs(((w**2 - w*b)/(w**2))-overlap))

        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx + 1
        elif H > W:
            Hd = idx + 1

        self.regions = []
        for l in range(levels):

            wl = int(2 * w/(l+2));
            wl2 = int(wl / 2 - 1);

            b = 0 if not (l + Wd) else ((W - wl) / (l + Wd))
            cenW = np.asarray(np.floor(wl2 + np.asarray(range(l+Wd+1)) * b), dtype=np.int32) - wl2
            b = 0 if not (l + Hd) else ((H - wl) / (l + Hd))
            cenH = np.asarray(np.floor(wl2 + np.asarray(range(l+Hd+1)) * b), dtype=np.int32) - wl2

            for i in cenH:
                for j in cenW:
                    if i >= W or j >= H:
                        continue
                    ie = i+wl
                    je = j+wl
                    if ie >= W:
                        ie = W
                    if je >= H:
                        je = H
                    if ie - i < 1 or je - j < 1:
                        continue
                    self.regions.append((i,j,ie,je))

        if verbose:
            print('RMAC regions = %s' % self.regions)
    # x [B,D,W,H]
    def rmac(self, x):
        y = []
        m_max = AdaptiveMaxPool2d((1, 1))
        m_mean = AdaptiveAvgPool2d((1, 1))
        for r in self.regions:
            x_sliced = x[:, :, r[1]:r[3], r[0]:r[2]]
            if self.power is None:
                x_maxed = m_max(x_sliced) # x_maxed [B,K]
            else:
                x_maxed = m_mean((x_sliced ** self.power)) ** (1.0 / self.power)
                x_maxed = torch.pow(m_mean((torch.pow(x_sliced, self.power))),(1.0 / self.power))
            y.append(x_maxed.squeeze(-1).squeeze(-1))
        # y list(N) N [B,K]
        y = torch.stack(y, dim=0)  # y [N,B,K]
        y = y.transpose(0,1)  # y [B,N,K]

        if self.norm:
            y = F.normalize(y, p=2, dim=-1)  # y [B,N,K]

        m_max = AdaptiveMaxPool2d((1, None))
        if self.sum_fm:
            y = AdaptiveMaxPool2d((1, None))(y)  # y [B,K]
            y = y.squeeze(1)
        return y

