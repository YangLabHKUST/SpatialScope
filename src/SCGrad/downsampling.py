import torch

from SCGrad.base import BaseModule
from SCGrad.interpolation import InterpolationBlock, Upsample, Downsample
from SCGrad.nn_layers import Conv1dWithInitialization


class ConvolutionBlock(BaseModule):
    def __init__(self, in_channels, out_channels, dilation):
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )
    
    def forward(self, x):
        outputs = self.leaky_relu(x)
        outputs = self.convolution(outputs)
        return outputs


class DownsamplingBlock(BaseModule):
    def __init__(self, in_channels, out_channels, factor, dilations, downsampling = True, remain_dim = None):
        super(DownsamplingBlock, self).__init__()
        if downsampling:
            sampling_layer1 = Downsample(in_channels = in_channels, scale_factor = factor)
            sampling_layer2 = Downsample(in_channels = out_channels, scale_factor = factor)
        else:
            sampling_layer1 = Upsample(in_channels = in_channels, scale_factor = factor, remain_dim = remain_dim)
            sampling_layer2 = Upsample(in_channels = out_channels, scale_factor = factor, remain_dim = remain_dim)
            
            
        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]
        out_sizes = [out_channels for _ in range(len(in_sizes))]
        self.main_branch = torch.nn.Sequential(*([
#             InterpolationBlock(
#                 scale_factor=factor,
#                 mode='linear',
#                 align_corners=False,
#                 downsample=True
#             )
            sampling_layer1
        ] + [
            ConvolutionBlock(in_size, out_size, dilation)
            for in_size, out_size, dilation in zip(in_sizes, out_sizes, dilations)
        ]))
        self.residual_branch = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
#             InterpolationBlock(
#                 scale_factor=factor,
#                 mode='linear',
#                 align_corners=False,
#                 downsample=True
#             )
            sampling_layer2
        ])

    def forward(self, x):
        outputs = self.main_branch(x)
        outputs = outputs + self.residual_branch(x)
        return outputs

    
    
