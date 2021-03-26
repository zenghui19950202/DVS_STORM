#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/10/10
import torch
import torch.nn as nn
import heapq
# import torch.fft as fft
from self_supervised_learning_sr import forward_model
from self_supervised_learning_sr import processing_utils
from utils import Fourier
# from utils.common_utils import plot_single_tensor_image
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pattern, pattern_gt, mask,normalize = True, deconv = False):
        num_of_channels = pattern.size()[1]
        # mean_pattern = pattern.detach().mean()
        # mean_pattern_gt = pattern_gt.detach().mean()
        if normalize == False:
            mean_pattern = torch.tensor([1.0], device=pattern.device)
            mean_pattern_gt = torch.tensor([1.0], device=pattern.device)
        else:
            mean_pattern = pattern.mean()
            mean_pattern_gt = pattern_gt.mean()
        # mean_pattern = pattern.detach().max()
        # mean_pattern_gt = pattern.detach().max()
        if mean_pattern < 1e-20:
            pattern_nomalized = pattern / (mean_pattern + 1e-19)
        else:
            pattern_nomalized = pattern / mean_pattern
        if mean_pattern_gt < 1e-20:
            pattern_gt_nomalized = pattern_gt / (mean_pattern_gt + 1e-19)
        else:
            pattern_gt_nomalized = pattern_gt / mean_pattern_gt

        loss = torch.mean(torch.pow((pattern_nomalized - pattern_gt_nomalized) * mask, 2))
        return loss * num_of_channels

class MSE_loss_1(nn.Module):
    """try ro precisely calculate the max-min range of FFT frequency spectra, but the computation consumption is large"""
    def __init__(self):
        super().__init__()

    def forward(self, pattern, pattern_gt, mask):
        _,channels,image_sizex,image_sizey = pattern.shape
        loss = torch.tensor([0.],device = pattern.device)
        for i in range(channels):
            pattern_slice = pattern[:,i,:,:].squeeze()
            pattern_slice_no_bg = pattern_slice - pattern_slice.detach().mean()
            pattern_slice_no_bg_max_list = heapq.nlargest(10, pattern_slice_no_bg.detach().view(1,-1).squeeze())
            pattern_slice_no_bg_max = torch.tensor([0.],device =pattern_slice_no_bg.device)

            for i in pattern_slice_no_bg_max_list:
                pattern_slice_no_bg_max+= i
            pattern_slice_no_bg_max = pattern_slice_no_bg_max / len(pattern_slice_no_bg_max_list)

            if pattern_slice_no_bg_max < 1e-20:
                pattern_nomalized = pattern_slice_no_bg / (pattern_slice_no_bg_max + 1e-19)
            else:
                pattern_nomalized = pattern_slice_no_bg / pattern_slice_no_bg_max

            loss += torch.mean(torch.pow((pattern_nomalized - pattern_gt)* mask, 2))
        return loss


class MSE_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pattern, pattern_gt, mask, OTF, normalize=True, deconv=False):
        num_of_channels = pattern.size()[1]
        # mean_pattern = pattern.detach().mean()
        # mean_pattern_gt = pattern_gt.detach().mean()
        if normalize == False:
            mean_pattern = torch.tensor([1.0], device=pattern.device)
            mean_pattern_gt = torch.tensor([1.0], device=pattern.device)
        else:
            mean_pattern = pattern.mean()
            mean_pattern_gt = pattern_gt.mean()
        # mean_pattern = pattern.detach().max()
        # mean_pattern_gt = pattern.detach().max()
        if mean_pattern < 1e-20:
            pattern_nomalized = pattern / (mean_pattern + 1e-19)
        else:
            pattern_nomalized = pattern / mean_pattern
        if mean_pattern_gt < 1e-20:
            pattern_gt_nomalized = pattern_gt / (mean_pattern_gt + 1e-19)
        else:
            pattern_gt_nomalized = pattern_gt / mean_pattern_gt

        if deconv == True:
            loss = torch.mean(torch.pow((pattern_nomalized - pattern_gt_nomalized) * mask * (OTF / (OTF * OTF + 0.04)).unsqueeze(2), 2))
        else:
            loss = torch.mean(torch.pow((pattern_nomalized - pattern_gt_nomalized) * mask, 2))
        return loss * num_of_channels


class MSE_loss_normal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pattern, pattern_gt):
        loss = torch.tensor([0.],device = pattern.device)
        loss += torch.mean(torch.pow((pattern - pattern_gt), 2))
        return loss

def residual_peak_loss(image,pattern_parameters):
        notch_filter = processing_utils.notch_filter_generate(image, pattern_parameters,notch_radius = 3)
        image_complex = torch.stack(
            [image.squeeze(), torch.zeros_like(image).squeeze()], 2)
        residual_peak = forward_model.torch_2d_fftshift(torch.fft(image_complex,2)) * (1-notch_filter).unsqueeze(2)
        residual_peak_loss = torch.norm(residual_peak.abs(), p=2)
        return residual_peak_loss

def tv_loss_calculate(x, beta=1):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    if x.dim() == 4:
        pass
    elif x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    else:
        raise Exception('dim of inputted x must be 2 or 4')

    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1 ,:] + dw[:, :, :, :-1], beta))

def minus_loss_calculate(x):
    minus_x = torch.mean(abs(x) - x)

    return minus_x