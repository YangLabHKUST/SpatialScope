import torch
import torch.nn as nn

from SCGrad.base import BaseModule


class InterpolationBlock(BaseModule):
    def __init__(self, scale_factor, mode='linear', align_corners=False, downsample=False):
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        outputs = torch.nn.functional.interpolate(
            x,
            size=x.shape[-1] * self.scale_factor \
                if not self.downsample else x.shape[-1] // self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False
        )
        return outputs
    
    
class Downsample(nn.Module):
    def __init__(self, in_channels, scale_factor, with_conv = True):
        super().__init__()
        self.with_conv = with_conv
#         pad_num = scale_factor // 2
#         if (scale_factor % 2) == 0:
#             self.pad = (pad_num - 1, pad_num)
#         else:
#             self.pad = (pad_num, pad_num)
        
        
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=7,
                                        stride=scale_factor,
                                        padding=3)

    def forward(self, x):
        if self.with_conv:
#             pad = (2, 2)
#             x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
    
    
class Upsample(nn.Module):
    def __init__(self, in_channels, scale_factor, remain_dim, with_conv = True):
        super().__init__()
        self.with_conv = with_conv
        if remain_dim == None:
            remain_dim = 0
        if self.with_conv:
            self.conv = torch.nn.ConvTranspose1d(in_channels,
                                                 in_channels,
                                                 kernel_size=7,
                                                 stride=scale_factor,
                                                 padding=3,
                                                 output_padding = remain_dim)

    def forward(self, x):
#         x = torch.nn.functional.interpolate(
#             x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x



    