#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/6/16

import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torch.utils.data import DataLoader
from math import pi
import numpy as np
import cv2


# from Add_Sinpattern_OTF_noise_ByAugumentor import SinusoidalPattern

class SIM_data_load(data.Dataset):
    def __init__(self,
                 directory_data_file,
                 data_mode='SIM_and_LR_images'
                 ,normalize=True
                 ):
        # data_dict=[]
        # with open(directory_data_file,'r',encoding='utf8') as json_file:
        #     for line in json_file.readlines():
        #         dic = json.loads(line)
        #         data_dict.append(dic)
        #     print(data_dict)
        with open(directory_data_file, 'r') as txtFile:
            self.content = txtFile.readlines()

        self.data_mode = data_mode
        self.normalize = normalize

    def __getitem__(self, index):
        txt_line = self.content[index]
        SIM_data_directory = txt_line.split()[0]
        image_number = int(txt_line.split()[1])
        image_format = txt_line.split()[2]
        self.SIM_data_directory = SIM_data_directory
        self.image_number = image_number
        self.image_format = image_format

        LR_image_directoty = SIM_data_directory \
                             + "_LR_" \
                             + '.' + image_format

        HR_image_directoty = SIM_data_directory \
                             + "_SR_" \
                             + '.' + image_format

        if image_format == 'tif' or image_format == 'tiff':
            HR_image_np = cv2.imread(HR_image_directoty, -1)/1.0
            LR_image_np = cv2.imread(LR_image_directoty, -1)/1.0
            image_size = [LR_image_np.shape[0], LR_image_np.shape[1]]
            self.image_size = image_size
            HR_image_tensor = torch.from_numpy(HR_image_np)
            LR_image_tensor = torch.from_numpy(LR_image_np)

            SIM_image_data = torch.zeros(image_number + 1, image_size[0], image_size[1])

            for i in range(image_number):
                SIM_data_image_directoty = SIM_data_directory \
                                           + "_Speckle_SIM_data(" \
                                           + str(i + 1) \
                                           + ")_" \
                                           + '.' + image_format
                SIM_image_np = cv2.imread(SIM_data_image_directoty, -1)/1.0
                SIM_image_tensor = torch.from_numpy(SIM_image_np)

                SIM_image_data[i, :, :] = SIM_image_tensor
                # SIM_image_data[i, :, :] = torch.zeros_like(SIMdata_normalized_image_tensor[0,:,:])
        else:
            HR_image_PIL = Image.open(HR_image_directoty)
            LR_image_PIL = Image.open(LR_image_directoty)
            HR_image_PIL = HR_image_PIL.convert('RGB')
            LR_image_PIL = LR_image_PIL.convert('RGB')

            if len(HR_image_PIL.size) == 2:
                image_size = [HR_image_PIL.size[0], HR_image_PIL.size[1]]
            elif len(HR_image_PIL.size) == 3:
                image_size = [HR_image_PIL.size[1], HR_image_PIL.size[2]]
            self.image_size = image_size

            if self.normalize == True:
                transform = transforms.Compose(
                    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor()])
            self.transform = transform

            HR_normalized_image_tensor = transform(HR_image_PIL)[0, :, :]
            LR_normalized_image_tensor = transform(LR_image_PIL)[0, :, :]
            LR_image_tensor = LR_normalized_image_tensor
            HR_image_tensor = HR_normalized_image_tensor
            SIM_image_data = torch.zeros(image_number + 1, image_size[0], image_size[1])

            for i in range(image_number):
                SIM_data_image_directoty = SIM_data_directory \
                                           + "_Speckle_SIM_data(" \
                                           + str(i + 1) \
                                           + ")_" \
                                           + '.' + image_format
                SIM_image_PIL = Image.open(SIM_data_image_directoty)
                SIM_image_PIL = SIM_image_PIL.convert('RGB')
                SIMdata_normalized_image_tensor = transform(SIM_image_PIL)
                SIM_image_data[i, :, :] = SIMdata_normalized_image_tensor[0, :, :]
                # SIM_image_data[i, :, :] = torch.zeros_like(SIMdata_normalized_image_tensor[0,:,:])

        if self.data_mode == 'SIM_and_LR_images':
            SIM_image_data[image_number, :, :] = LR_image_tensor  # directly use the LR image as input
        elif self.data_mode == 'SIM_and_sum_images':
            appro_widefield_image = SIM_image_data.mean(0)
            SIM_image_data[image_number, :,
            :] = appro_widefield_image  # add one layer of approximately wide field image(mean of all the SIM)
        else:
            SIM_image_data = SIM_image_data.narrow(0, 0, image_number)

        if self.image_number == 3:
            SIM_image_data = torch.cat((SIM_image_data, LR_image_tensor.unsqueeze(0)), 0)
        return SIM_image_data, torch.stack((HR_image_tensor, LR_image_tensor), 2)

    def __len__(self):
        return len(self.content)


class SIM_pattern_load(SIM_data_load):
    def __getitem__(self, index):
        super(SIM_pattern_load, self).__getitem__(index)

        SIM_pattern_data = torch.zeros(self.image_number , self.image_size[0], self.image_size[1])

        if self.image_format == 'tif' or self.image_format == 'tiff':
            for i in range(self.image_number):
                SIM_data_image_directoty = self.SIM_data_directory \
                                           + "_Speckle_SIM_pattern(" \
                                           + str(i + 1) \
                                           + ")_" \
                                           + '.' + self.image_format
                SIM_image_np = cv2.imread(SIM_data_image_directoty, -1)/1.0
                SIM_image_tensor = torch.from_numpy(SIM_image_np)
                SIM_pattern_data[i, :, :] = SIM_image_tensor
        else:
            for i in range(self.image_number):
                SIM_data_image_directoty = self.SIM_data_directory \
                                           + "_Speckle_SIM_pattern(" \
                                           + str(i + 1) \
                                           + ")_" \
                                           + '.' + self.image_format
                SIM_pattern_PIL = Image.open(SIM_data_image_directoty)
                SIM_pattern_PIL = SIM_pattern_PIL.convert('RGB')
                SIM_pattern_normalized_image_tensor = self.transform(SIM_pattern_PIL)
                SIM_pattern_data[i, :, :] = SIM_pattern_normalized_image_tensor[0, :, :]
                # SIM_image_data[i, :, :] = torch.zeros_like(SIMdata_normalized_image_tensor[0,:,:])
        if self.image_number == 3:
            even_illunimation = torch.ones(1,self.image_size[0], self.image_size[1])
            SIM_pattern_data = torch.cat((SIM_pattern_data, even_illunimation), 0)
        return SIM_pattern_data

    def __len__(self):
        return len(self.content)


if __name__ == '__main__':
    directory_json_file = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown\\test\directories_of_images.json"
    directory_txt_file = 'D:\DataSet\DIV2K\DIV2K/SIMdata_SR_train.txt'

    SIM_dataset = SIM_pattern_load(directory_txt_file)
    a = SIM_dataset[0]
    a = a * 0.5 + 0.5
    a_PIL = transforms.ToPILImage()(a[0, :, :]).convert('RGB')
    a_PIL.show()
    # b = b * 0.5 + 0.5
    # b_PIL = transforms.ToPILImage()(b[:, :, 0]).convert('RGB')
    SIM_data_loader = DataLoader(SIM_dataset, batch_size=4, shuffle=True)
