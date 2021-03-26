#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author  ：zh , Date  ：2021/3/17 10:01

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import sys
import os
import torchsummary

sys.path.append(os.path.dirname(os.getcwd()))

from models.SNN_model import *
from simulation_data_generation.load_dataset import STORM_DVS
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import load_configuration_parameters
from simulation_data_generation import Generate_DVS_STORM_data
import uuid

data_path = "E:\PHD\DVS_STORM_SOFI\DVS\DVS_data/"
save_path = ''
from models import ANN_model


def train(args, model, device, train_loader, optimizer, epoch, writer, criterion, save_id):
    model.train()
    print('save_id:' + str(save_id))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        output = model(data).squeeze()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalars('scalar/loss', {'train_loss': loss}, epoch + 1)

    if epoch % 5 == 0:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        state = {
            'net': model.state_dict(),
            'train_loss': loss,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/' + str(save_id) + 'Decoder_4x_SNN' + '.t7')


def test(args, model, device, test_loader, epoch, writer, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            output = model(data)
            output = output.squeeze()
            test_loss += criterion(output, target)
            # test_loss += weighted_MSE_loss(output, target)

    test_loss /= len(test_loader)
    writer.add_scalars('scalar/loss', {'test_loss': test_loss}, epoch + 1)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def weighted_MSE_loss(predict, GT):
    # predict = predict/predict.max()
    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    fluorophore_density = data_generation_parameters['fluorophore_density']
    loss = torch.mean(torch.pow((predict - GT) * GT, 2) / fluorophore_density + torch.pow((predict - GT) * (1 - GT), 2))
    L1_loss = nn.L1Loss()
    loss+= L1_loss(predict,torch.zeros_like(predict))
    return loss


class psf_weighted_loss(nn.Module):
    def __init__(self):
        super(psf_weighted_loss, self).__init__()
        self.simulator = Generate_DVS_STORM_data.STORM_DVS_simulator()

    def forward(self, predict, GT):
        mask = self.simulator.batch_image_OTF_filter(GT)
        loc = mask > 0
        mask[loc] = 1
        L1_loss = nn.L1Loss()
        data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
        fluorophore_density = data_generation_parameters['fluorophore_density']
        loss = torch.mean(
            torch.pow((predict - GT) * mask, 2) / fluorophore_density + torch.pow((predict - GT) * (1 - mask), 2))

        loss += L1_loss(predict, torch.zeros_like(predict))
        nn.Softmax

        return loss


class psf_loss(nn.Module):
    def __init__(self,device):
        super(psf_loss, self).__init__()
        self.simulator = Generate_DVS_STORM_data.STORM_DVS_simulator()
        OTF = self.simulator.OTF_form(fc_ratio= 1)
        self.psf_conv = Generate_DVS_STORM_data.psf_conv_generator(self.simulator.psf_form(OTF),device)

    def forward(self, predict, GT):
        # GT_diffractive = abs(self.simulator.batch_image_OTF_filter(GT))
        # predict_diffractive = abs(self.simulator.batch_image_OTF_filter(predict))
        GT_diffractive = self.psf_conv(GT)
        predict_diffractive = self.psf_conv(predict)
        L1_loss = nn.L1Loss()
        loss = torch.mean(torch.pow((predict_diffractive - GT_diffractive), 2))
        loss += L1_loss(predict, torch.zeros_like(predict))

        return loss

class MSE_and_L1_loss(nn.Module):
    def __init__(self):
        super(MSE_and_L1_loss, self).__init__()
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
        self.fluorophore_density = data_generation_parameters['fluorophore_density']

    def forward(self, predict, GT):
        GT = GT / self.fluorophore_density
        loss = self.MSE_loss(predict,GT)
        loss += self.L1_loss(predict, torch.zeros_like(predict))

        return loss


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=48, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=48, metavar='N',
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
    parser.add_argument('--save-net', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter('./summaries/cifar10')

    data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
    data_path = data_generation_parameters['output_directory']

    train_dataset = STORM_DVS(train=True,  win=100, path=data_path, net_model='ANN')
    test_dataset = STORM_DVS(train=False,  win=100, path=data_path, net_model='ANN')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    device = torch.device("cuda:6" if use_cuda else "cpu")
    net = ANN_model.Unet_4x_ANN()
    # net = nn.DataParallel(net, device_ids=[2, 3])
    net = net.to(device)
    net.apply(init_weights)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    criterion = psf_loss(device)
    # criterion = psf_weighted_loss()
    criterion = MSE_and_L1_loss()
    # criterion = nn.CrossEntropyLoss()
    save_id = uuid.uuid4()
    for epoch in range(1, args.epochs + 1):
        train(args, net, device, train_loader, optimizer, epoch, writer, criterion, save_id)
        test(args, net, device, test_loader, epoch, writer, criterion)

    writer.close()


if __name__ == '__main__':
    # net = torchvision.models.resnet18()
    # torchsummary.summary(net,input_size=[(3,128,128)],batch_size=1,device='cpu')
    main()
