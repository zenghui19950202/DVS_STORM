#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author  ：zh , Date  ：2021/3/26 12:30

from torchvision import transforms
from torch.utils import data
from PIL import Image
from utils import common_utils
from simulation_data_generation import Generate_DVS_STORM_data
from utils import load_configuration_parameters
import torch

sample_data_directoty = '/data/zh/DVS_STORM_Dataset_SNN_8x/target.tif'

output_directory =  '/data/zh/DVS_STORM_sample_data/'

sample_PIL = Image.open(sample_data_directoty)

if len(sample_PIL.size) == 2:
    image_size = [sample_PIL.size[0], sample_PIL.size[1]]
elif len(sample_PIL.size) == 3:
    image_size = [sample_PIL.size[1], sample_PIL.size[2]]

transform = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     ])

sample_image_tensor = transform(sample_PIL)[0, :, :]
sample_image_tensor = 1 - sample_image_tensor
loc = sample_image_tensor == sample_image_tensor.min()
sample_image_tensor[loc] = 0


data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
fluorophore_density = data_generation_parameters['fluorophore_density']
STORM_frame_num = round(1 / fluorophore_density)
STORM_frame_num = 10

for i in range(STORM_frame_num):
    simulator = Generate_DVS_STORM_data.STORM_DVS_simulator()
    single_STORM_image = simulator.generate_single_frame(str(i + 1), sample_image_tensor)
    simulator.generate_event_data_from_single_frame(single_STORM_image, str(i + 1),output_directory)
    print(i)
