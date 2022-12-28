import numpy as np
import os
import sys


import torch
import torch.nn as nn

from typing import List, Tuple, Union, Callable


from SCGrad.base import BaseModule
from SCGrad.nn_layers import Conv1dWithInitialization
from SCGrad.upsampling import UpsamplingBlock as UBlock
from SCGrad.downsampling import DownsamplingBlock as DBlock
from SCGrad.linear_modulation import FeatureWiseLinearModulation_NoScalarCond as FiLM


def cal_final_dim(input_dim, factor):
    output_dim = (((input_dim + 4) - 7) // factor) + 1
    return output_dim

def remain_dim(input_dim, factor):
    remain_dim = ((input_dim + 4) - 7) % factor
    return remain_dim

def cal_final_updim(output_dim, factor):
    output_dim_ori = output_dim
    output_dim = int((output_dim - 1) / factor + 1)
    output_padding = output_dim_ori - ((output_dim - 1) * factor+1)
    return output_dim, output_padding


class SCGradNN(BaseModule):
    def __init__(self,
        input_dim1: int,
        down_block_dim: list = [32,  128, 256, 512],
        factors: list = [3,4,5,1],
        downsampling_dilations: list = [
            [1,2,4],
            [1,2,4],
            [1,2,4],
            [1,2,4],
            [1,2,4]
        ],
        upsampling_dilations: list = [
            [1,2,1,2],
            [1,2,1,2],
            [1,2,1,2],
            [1,2,1,2],
            [1,2,1,2]
        ],
        seed=182822,
    ):
        super(SCGradNN, self).__init__()
        
        self.input_dim1 = input_dim1
        self.down_block_dim = down_block_dim
        self.factors = factors
        self.downsampling_dilations = [[1,2,4]] * len(self.down_block_dim)
        self.upsampling_dilations = [[1,2,1,2]] * len(self.down_block_dim)
        
        
        cal_factor = self.factors[:-1]
        output_dim = self.input_dim1
        self.output_padding = []
        for fac in cal_factor:
            output_dim, out_padding = cal_final_updim(output_dim, fac)
            self.output_padding.append(out_padding)
            
            
        # U_net left stream
        self.left_ublock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=self.down_block_dim[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.left_ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                downsampling = True
            ) for in_size, out_size, factor, dilations in zip(
                self.down_block_dim,
                self.down_block_dim[1:] + [self.down_block_dim[-1]],
                self.factors,
                self.upsampling_dilations
            )
        ])
        

        # Building downsampling branch (starting from signal)
        self.left_dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=down_block_dim[0],
            kernel_size=5,
            stride=1,
            padding=2
        )
        # downsampling_in_sizes = [config.model_config.downsampling_preconv_out_channels] \
            # + config.model_config.downsampling_out_channels[:-1]
        self.left_dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                self.down_block_dim,
                self.down_block_dim[1:] + [self.down_block_dim[-1]],
                self.factors,
                self.downsampling_dilations
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = self.down_block_dim[1:] + [self.down_block_dim[-1]]
        film_out_sizes = self.down_block_dim[1:] + [self.down_block_dim[-1]]
        film_factors = [1] + self.factors[1:][::-1]
        self.left_films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.product(film_factors[:i+1])  # for proper positional encodings initialization
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])
        
        ###################################################################################################
        # U_net right stream
        
        self.right_ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                remain_dim = remain_dim
            ) for in_size, out_size, factor, dilations,remain_dim in zip(
                (np.array(self.down_block_dim[1:]) * 2)[::-1],
                self.down_block_dim[:-1][::-1],
                self.factors[:-1][::-1],
                self.upsampling_dilations[:-1],
                self.output_padding[::-1]
            )
        ])
        
        
        self.right_ublock_postconv = Conv1dWithInitialization(
            in_channels=self.down_block_dim[0],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

     
        # downsampling_in_sizes = [config.model_config.downsampling_preconv_out_channels] \
            # + config.model_config.downsampling_out_channels[:-1]
        self.right_dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                downsampling = False,
                remain_dim = remain_dim
            ) for in_size, out_size, factor, dilations, remain_dim in zip(
                (np.array(self.down_block_dim[1:]) * 2)[::-1],
                self.down_block_dim[:-1][::-1],
                self.factors[:-1][::-1],
                self.downsampling_dilations[:-1],
                self.output_padding[::-1]
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = self.down_block_dim[:-1][::-1]
        film_out_sizes = self.down_block_dim[:-1][::-1]
        film_factors = [1] + self.factors[1:][::-1]
        self.right_films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.product(film_factors[:i+1])  # for proper positional encodings initialization
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])
        
        
        

    def forward(self, x_sep, mu, noise_level):
        """
        Computes forward pass of neural network.
        :param x_sep (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_x_sep, T//hop_length]
        :param mu (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        x_sep = x_sep.unsqueeze(1)
        assert len(x_sep.shape) == 3  # B, 1, T
        mu = mu.unsqueeze(1)
        assert len(mu.shape) == 3  # B, 1, T
        
        
        # left stream
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        hs_u, hs_d = [], []
        left_dblock_outputs = self.left_dblock_preconv(mu)
        for dblock, film in zip(self.left_dblocks, self.left_films):
            left_dblock_outputs = dblock(left_dblock_outputs)
            scale, shift = film(x=left_dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
            hs_d.append(left_dblock_outputs)
        # statistics = statistics[::-1]
        
        left_ublock_outputs = self.left_ublock_preconv(x_sep)
        for i, ublock in enumerate(self.left_ublocks):
            scale, shift = statistics[i]
            left_ublock_outputs = ublock(x=left_ublock_outputs, scale=scale, shift=shift)
            hs_u.append(left_ublock_outputs)
        # outputs = self.ublock_postconv(ublock_outputs)
        
        _, _ = hs_u.pop(), hs_d.pop()
        
        ## right stream
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        right_dblock_outputs = left_dblock_outputs
        for dblock, film in zip(self.right_dblocks, self.right_films):
            right_dblock_outputs = dblock(torch.cat([right_dblock_outputs, hs_d.pop()], dim=1))
            scale, shift = film(x=right_dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        
        # Upsampling stream
        # ublock_outputs = self.ublock_preconv(x_sep)
        right_ublock_outputs = left_ublock_outputs
        for i, ublock in enumerate(self.right_ublocks):
            scale, shift = statistics[i]
            right_ublock_outputs = ublock(x=torch.cat([right_ublock_outputs, hs_u.pop()], dim = 1), scale=scale, shift=shift)
        outputs = self.right_ublock_postconv(right_ublock_outputs)
        return outputs.squeeze(1)

    
