#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author  ：zh , Date  ：2021/3/15 22:55
import torch
# from __future__ import print_function
from models import SNN_model
import argparse
from models.SNN_model import *
from simulation_data_generation.load_dataset import STORM_DVS
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import load_configuration_parameters
import matplotlib.pyplot as plt
from utils import common_utils
from models import ANN_model
import os


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter('./summaries/cifar10')

    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    data_path = data_generation_parameters['output_directory']

    train_dataset = STORM_DVS(train=True,  win=100, path=data_path, net_model='ANN', Normalize = 'True')
    test_dataset = STORM_DVS(train=False,  win=100, path=data_path, net_model='ANN', Normalize = 'True')

    # train_dataset = STORM_DVS(train=True,  win=100, path=data_path, net_model='SNN', Normalize = 'True')
    # test_dataset = STORM_DVS(train=False,  win=100, path=data_path, net_model='SNN', Normalize = 'True')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # data_path = "E:\PHD\DVS_STORM_SOFI\DVS\DVS_data/"
    # net = model.enconder_decoder_4x_SNN().to(device)
    # net = ANN_model.resnet18().to(device)
    net = ANN_model.Unet_4x_ANN().to(device)
    # net = SNN_model.Unet_8x_SNN().to(device)

    state = torch.load('./checkpoint/6c18065b-f314-4a09-b809-5a0bb14be388' + 'Decoder_4x_SNN' + '.t7')
    net.load_state_dict(state['net'])

    # from collections import OrderedDict
    # def multi_GPU_net_load(model, check):
    #     new_state = OrderedDict()
    #     for layer_multi_GPU, name in state['net'].items():
    #         layer_single_gpu = layer_multi_GPU[7:]
    #         new_state[layer_single_gpu] = name
    #     model.load_state_dict(new_state)
    #     return model
    #
    # net = multi_GPU_net_load(net,state)


    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        output = net(data)
        common_utils.plot_single_tensor_image(data[0,:,:,:].squeeze()) # for ANN
        # common_utils.plot_single_tensor_image(data[:, :, :, :, 0].squeeze()) # for SNN
        common_utils.plot_single_tensor_image(output.squeeze())
        common_utils.plot_single_tensor_image(target.squeeze())

        if batch_idx >0:
            break
if __name__ == '__main__':
    main()
