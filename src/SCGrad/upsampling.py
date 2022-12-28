import torch

from SCGrad.base import BaseModule
from SCGrad.linear_modulation import FeatureWiseAffine
from SCGrad.interpolation import InterpolationBlock, Upsample, Downsample
from SCGrad.nn_layers import Conv1dWithInitialization



class BasicModulationBlock(BaseModule):
    """
    Linear modulation part of UBlock, represented by sequence of the following layers:
        - Feature-wise Affine
        - LReLU
        - 3x1 Conv
    """
    def __init__(self, n_channels, dilation):
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine(n_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        outputs = self.leaky_relu(outputs)
        outputs = self.convolution(outputs)
        return outputs


class UpsamplingBlock(BaseModule):
    def __init__(self, in_channels, out_channels, factor, dilations, downsampling = False, remain_dim = None):
        super(UpsamplingBlock, self).__init__()
        if downsampling:
            sampling_layer1 = Downsample(in_channels = in_channels, scale_factor = factor)
            sampling_layer2 = Downsample(in_channels = out_channels, scale_factor = factor)
        else:
            sampling_layer1 = Upsample(in_channels = in_channels, scale_factor = factor, remain_dim = remain_dim)
            sampling_layer2 = Upsample(in_channels = out_channels, scale_factor = factor, remain_dim = remain_dim)
        self.first_block_main_branch = torch.nn.ModuleDict({
            'upsampling': torch.nn.Sequential(*[
                torch.nn.LeakyReLU(0.2),
#                 InterpolationBlock(
#                     scale_factor=factor,
#                     mode='linear',
#                     align_corners=False
#                 ),
                
                sampling_layer1,
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilations[0],
                    dilation=dilations[0]
                )
            ]),
            'modulation': BasicModulationBlock(
                out_channels, dilation=dilations[1]
            )
        })
        self.first_block_residual_branch = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
#             InterpolationBlock(
#                 scale_factor=factor,
#                 mode='linear',
#                 align_corners=False
#             )
            sampling_layer2
        ])
        self.second_block_main_branch = torch.nn.ModuleDict({
            f'modulation_{idx}': BasicModulationBlock(
                out_channels, dilation=dilations[2 + idx]
            ) for idx in range(2)
        })

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch['upsampling'](x)
        outputs = self.first_block_main_branch['modulation'](outputs, scale, shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second residual block
        residual = self.second_block_main_branch['modulation_0'](outputs, scale, shift)
        outputs = outputs + self.second_block_main_branch['modulation_1'](residual, scale, shift)
        return outputs