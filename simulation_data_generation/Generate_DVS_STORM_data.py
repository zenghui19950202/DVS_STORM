#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/3/10

import torch
import math
import numpy as np
import random
from configparser import ConfigParser
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils import load_configuration_parameters
from utils import common_utils
import torch.fft as fft
import h5py
from v2e_utils import all_images, read_image, \
    video_writer, checkAddSuffix
import os
from output.aedat2_output import AEDat2Output
from output.ae_text_output import DVSTextOutput
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def torch_2d_fftshift(image):
    image = image.squeeze()
    num = image.dim() - 2
    shift = [image.shape[ax + num] // 2 for ax in range(2)]
    image_shift = torch.roll(image, shift, [num, num + 1])
    return image_shift


def torch_2d_ifftshift(image):
    image = image.squeeze()
    num = image.dim() - 1
    shift = [-(image.shape[num - ax] // 2) for ax in range(2)]
    image_shift = torch.roll(image, shift, [num, num - 1])
    return image_shift


class STORM_DVS_simulator:
    """
    This class is used to add Sinusoidal Pattern on images.
    """

    def __init__(self,
                 train=True,
                 pos_thres=0.09,
                 neg_thres=0.05,
                 sigma_thres=0.03,
                 cutoff_hz=0,
                 leak_rate_hz=0.1,
                 refractory_period_s=0,  # todo not yet modeled
                 shot_noise_rate_hz=0.001,  # rate in hz of temporal noise events
                 #  seed=42,
                 seed=0,
                 output_folder: str = None,
                 dvs_h5: str = None,
                 dvs_aedat2: str = None,
                 dvs_text: str = True,
                 # change as you like to see 'baseLogFrame',
                 # 'lpLogFrame', 'diff_frame'
                 show_dvs_model_state: str = None
                 # dvs_rosbag=None
                 ):  # unit: nm
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`percentage_area` to control the area
        of the image to crop in terms of its percentage of the original image,
        and a :attr:`centre` parameter toggle whether a random area or the
        centre of the images should be cropped.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        data_generation_parameters = load_configuration_parameters.load_data_generation_config_paras()
        # config = ConfigParser()
        # config.read('../configuration.ini')
        output_directory = data_generation_parameters['output_directory']
        self.Magnification = data_generation_parameters['Magnification']
        self.PixelSizeOfCCD = data_generation_parameters['PixelSizeOfCCD']
        self.EmWaveLength = data_generation_parameters['EmWaveLength']
        self.NA = data_generation_parameters['NA']
        self.image_size = data_generation_parameters['image_size']
        self.fluorophore_density = data_generation_parameters['fluorophore_density']
        self.downsample_rate = data_generation_parameters['downsample_rate']
        self.parallel_frames = data_generation_parameters['parallel_frames']
        # self.image_size = data_generation_parameters['image_size']
        self.f_cutoff = 1 / 0.61 * self.NA / self.EmWaveLength  # The coherent cutoff frequency Rayleigh criterion

        self.PixelSize = self.PixelSizeOfCCD / self.Magnification / self.downsample_rate
        self.delta_x = self.PixelSize  # xy方向的空域像素间隔，单位m
        self.delta_y = self.PixelSize
        self.delta_fx = 1 / self.image_size / self.delta_x  # xy方向的频域像素间隔，单位m ^ -1
        self.delta_fy = 1 / self.image_size / self.delta_y
        self.xx, self.yy, self.fx, self.fy = self.GridGenerate(grid_mode='real')

        self.f_grid = pow((self.fx ** 2 + self.fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )

        self.OTF = self.OTF_form()
        self.OTF_padding = self.padding_OTF_generate()
        # self.CTF = self.CTF_form()

        self.pos_thres = pos_thres
        self.time_window = 100  # unit: us
        self.output_folder = output_directory

        self.num_events_total = 0

        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.output_width = None
        self.output_height = None  # set on first frame
        self.show_input = show_dvs_model_state
        if seed > 0:
            np.random.seed(seed)

        self.dvs_h5 = dvs_h5
        self.dvs_aedat2 = dvs_aedat2
        self.dvs_text = dvs_text
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0
        self.train = train

    def generate_frame_batch(self):
        xx, yy, _, _ = self.GridGenerate(grid_mode='real')

        OTF = self.OTF
        random_distribution = torch.rand([self.parallel_frames, self.image_size, self.image_size])
        fluorophore_loc = random_distribution < self.fluorophore_density
        fluorophore_GT = torch.zeros_like(random_distribution)
        fluorophore_GT[fluorophore_loc] = 1

        fluorophore_diffractive_spectrum = torch_2d_fftshift(fft.fftn(fluorophore_GT, dim=[1, 2])) * OTF.unsqueeze(0)
        fluorophore_diffractive_image = abs(fft.ifftn(torch_2d_ifftshift(fluorophore_diffractive_spectrum), dim=[2, 1]))
        common_utils.plot_single_tensor_image(fluorophore_diffractive_image[0, :, :])
        # image_size_real = self.image_size / self.downsample_rate
        AvgPool_operation = nn.AvgPool2d(kernel_size=self.downsample_rate, stride=self.downsample_rate)
        fluorophore_image_in_camera = AvgPool_operation(fluorophore_diffractive_image.unsqueeze(0)).squeeze()

        return fluorophore_GT, fluorophore_image_in_camera

    def generate_single_frame(self, output_num, input_image = None):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        padding_size = 20
        _, _, fx, fy = self.GridGenerate(self.image_size + 2 * padding_size, grid_mode='real')
        f_grid = pow((fx ** 2 + fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )

        OTF_padding = self.OTF_form(f_grid)
        random_distribution = torch.rand([self.image_size, self.image_size])
        if input_image == None:
            input_image = torch.ones_like(random_distribution)
        random_distribution = random_distribution * input_image
        threhold = self.fluorophore_density
        fluorophore_num = round(self.fluorophore_density * self.image_size ** 2)
        fluorophore_loc = random_distribution < threhold

        while fluorophore_loc.sum() < fluorophore_num:
            threhold *= 2
            fluorophore_loc = random_distribution < threhold
        fluorophore_GT = torch.zeros_like(random_distribution)
        fluorophore_GT[fluorophore_loc] = 1
        ZeroPad_operation = nn.ZeroPad2d(20)
        fluorophore_GT_padding = ZeroPad_operation(fluorophore_GT)  # 直接进行频域OTF滤波，会有边缘信息的串扰，用zero_padding方法去除
        fluorophore_padding_diffractive_spectrum = torch_2d_fftshift(
            fft.fftn(fluorophore_GT_padding, dim=[0, 1])) * OTF_padding
        fluorophore_padding_diffractive_image = abs(
            fft.ifftn(torch_2d_ifftshift(fluorophore_padding_diffractive_spectrum), dim=[1, 0]))
        fluorophore_diffractive_image = fluorophore_padding_diffractive_image[padding_size:-padding_size,
                                        padding_size:-padding_size]
        # print(fluorophore_loc.sum())
        # common_utils.plot_single_tensor_image(fluorophore_diffractive_image)
        # image_size_real = self.image_size / self.downsample_rate
        AvgPool_operation = nn.AvgPool2d(kernel_size=self.downsample_rate, stride=self.downsample_rate)
        fluorophore_image_in_camera = AvgPool_operation(
            fluorophore_diffractive_image.unsqueeze(0).unsqueeze(0)).squeeze()
        # np.save(os.path.join(self.output_folder, output_num + '_label'), fluorophore_GT.numpy())

        fluorophore_GT_loc = (fluorophore_GT.numpy() == 1)
        fluorophore_GT_loc_xy = np.where(fluorophore_GT_loc)

        x = fluorophore_GT_loc_xy[0].astype(np.int32)
        y = fluorophore_GT_loc_xy[1].astype(np.int32)

        label_file_dir = os.path.join(self.output_folder, output_num + '_label.txt')
        label_file = open(label_file_dir, 'w')
        for i in range(len(x)):
            label_file.write('{} {}\n'.format(x[i], y[i]))  # todo there must be vector way
        label_file.close()

        return fluorophore_image_in_camera

    def generate_event_data_from_single_frame(self, fluorophore_image_in_camera, out_put_name,output_directory = None):
        # print(fluorophore_image_in_camera.max())

        fluorophore_image_in_camera = fluorophore_image_in_camera / (fluorophore_image_in_camera.max()+0) * 90 + 110
        background_matrix = torch.zeros_like(fluorophore_image_in_camera).numpy() + 110
        fluorophore_image_in_camera = fluorophore_image_in_camera.numpy()
        diff_image = self.lin_log(fluorophore_image_in_camera) - self.lin_log(background_matrix)
        # 是否需要用多个脉冲来表示强变化
        evts_frame = diff_image // self.pos_thres
        evts_frame = evts_frame.astype(int)
        # compute number of times to pass over array to compute
        num_iters = int(evts_frame.max())
        configuration_file_dir = os.path.join(self.output_folder, 'configuration.txt')
        configuration_file = open(configuration_file_dir, 'w')
        configuration_file.write(str(num_iters))

        events = []

        inten01 = (np.array(fluorophore_image_in_camera, float) + 20) / 275

        for i in range(num_iters):
            events_curr_iters = np.zeros((0, 4), dtype=np.float32)
            ts = self.time_window * (i + 1) / (num_iters + 1)
            ts = int(i)
            pos_cord = (
                    evts_frame >= i + 1)  # it must be >= because we need to make event for each iteration up to total # events for that pixel
            # generate events
            #  make a list of coordinates x,y addresses of events
            event_xy = np.where(pos_cord)
            num_events = event_xy[0].shape[0]

            self.num_events_total += num_events

            if num_events > 0:
                pos_events = np.hstack(
                    (np.ones((num_events, 1), dtype=np.float32) * ts,
                     event_xy[0][..., np.newaxis],
                     event_xy[1][..., np.newaxis],
                     np.ones((num_events, 1), dtype=np.float32) * 1))
            else:
                pos_events = np.zeros((0, 4), dtype=np.float32)

            events_tmp = pos_events

            # randomly order events to prevent bias to one corner
            #  if events_tmp.shape[0] != 0:
            #      np.random.shuffle(events_tmp)

            if num_events > 0:
                events_curr_iters = events_tmp
                #  events.append(events_tmp)

                if self.shot_noise_rate_hz > 0:
                    # NOISE: add temporal noise here by
                    # simple Poisson process that has a base noise rate
                    # self.shot_noise_rate_hz.
                    # If there is such noise event,
                    # then we output event from each such pixel

                    # the shot noise rate varies with intensity:
                    # for lowest intensity the rate rises to parameter.
                    # the noise is reduced by factor
                    # SHOT_NOISE_INTEN_FACTOR for brightest intensities
                    SHOT_NOISE_INTEN_FACTOR = 0.25
                    shotNoiseFactor = (
                                              (self.shot_noise_rate_hz / 2) * self.time_window / num_iters) * \
                                      ((SHOT_NOISE_INTEN_FACTOR - 1) * inten01 + 1)
                    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

                    rand01 = np.random.uniform(
                        size=diff_image.shape)  # draw samples

                    # probability for each pixel is
                    # dt*rate*nom_thres/actual_thres.
                    # That way, the smaller the threshold,
                    # the larger the rate
                    shotOnProbThisSample = shotNoiseFactor * np.divide(
                        self.pos_thres_nominal, self.pos_thres)
                    # array with True where ON noise event
                    shotOnCord = rand01 > (1 - shotOnProbThisSample)

                    shotOffProbThisSample = shotNoiseFactor * np.divide(
                        self.neg_thres_nominal, self.neg_thres)
                    # array with True where OFF noise event
                    shotOffCord = rand01 < shotOffProbThisSample

                    shotOnXy = np.where(shotOnCord)
                    shotOnCount = shotOnXy[0].shape[0]

                    shotOffXy = np.where(shotOffCord)
                    shotOffCount = shotOffXy[0].shape[0]

                    self.num_events_on += shotOnCount
                    self.num_events_off += shotOffCount
                    self.num_events_total += shotOnCount + shotOffCount
                    pos_thr = self.pos_thres if isinstance(
                        self.pos_thres, float) else self.pos_thres[shotOnCord]
                    neg_thr = self.neg_thres if isinstance(
                        self.neg_thres, float) else self.neg_thres[shotOffCord]
                    if shotOnCount > 0:
                        shotEvents = np.hstack(
                            (np.ones((shotOnCount, 1), dtype=np.float32) * ts,
                             shotOnXy[1][..., np.newaxis],
                             shotOnXy[0][..., np.newaxis],
                             np.ones((shotOnCount, 1), dtype=np.float32) * 1))
                        events_curr_iters = np.append(
                            events_curr_iters, shotEvents, axis=0)
                        #  events.append(shotEvents)
                    if shotOffCount > 0:
                        shotEvents = np.hstack(
                            (np.ones((shotOffCount, 1), dtype=np.float32) * ts,
                             shotOffXy[1][..., np.newaxis],
                             shotOffXy[0][..., np.newaxis],
                             np.ones((shotOffCount, 1), dtype=np.float32) * -1))
                        events_curr_iters = np.append(
                            events_curr_iters, shotEvents, axis=0)
                        #  events.append(shotEvents)
                        # self.baseLogFrame[shotOffCord] -= \
                        #     shotOffCord[shotOffCord] * \
                        #     neg_thr
                    # end temporal noise

            # shuffle and append to the events collectors
            # np.random.shuffle(events_curr_iters)
            events.append(events_curr_iters)

        if len(events) > 0:

            if self.output_folder:
                if self.dvs_h5:
                    path = os.path.join(self.output_folder, out_put_name)
                    path = checkAddSuffix(path, '.h5')
                    self.dvs_h5 = h5py.File(path, "w")
                    self.dvs_h5_dataset = self.dvs_h5.create_dataset(
                        name="events",
                        shape=(0, 4),
                        maxshape=(None, 4),
                        dtype="uint32",
                        compression="gzip")
                if self.dvs_aedat2:
                    path = os.path.join(self.output_folder, out_put_name)
                    path = checkAddSuffix(path, '.aedat')
                    self.dvs_aedat2 = AEDat2Output(path)
                if self.dvs_text:
                    if output_directory == None:
                        path = os.path.join(self.output_folder, out_put_name)
                        path = checkAddSuffix(path, '.txt')
                    else:
                        if not os.path.exists(output_directory):
                            os.makedirs(output_directory)
                        path = os.path.join(output_directory, out_put_name)
                        path = checkAddSuffix(path, '.txt')
                    self.dvs_text = DVSTextOutput(path)

            events = np.vstack(events)
            if self.dvs_h5 is not None:
                # convert data to uint32 (microsecs) format
                temp_events = np.copy(events)
                temp_events[:, 0] = temp_events[:, 0] * 1e6
                temp_events[temp_events[:, 3] == -1, 3] = 0
                temp_events = temp_events.astype(np.uint32)

                # save events
                self.dvs_h5_dataset.resize(
                    self.dvs_h5_dataset.shape[0] + temp_events.shape[0],
                    axis=0)

                self.dvs_h5_dataset[-temp_events.shape[0]:] = temp_events
                self.dvs_h5.flush()
            if self.dvs_aedat2 is not None:
                self.dvs_aedat2.appendEvents(events)
            if self.dvs_text is not None:
                self.dvs_text.appendEvents(events)
        self.dvs_text.close()
        if len(events) > 0:
            return events
        else:
            return None

    def lin_log(self, x, threshold=20):
        """
        linear mapping + logarithmic mapping.

        :param x: float or ndarray
            the input linear value in range 0-255
        :param threshold: float threshold 0-255
            the threshold for transisition from linear to log mapping

        :returns: the linlog value, in range 0-np.log(255) which is 0-5.55413

        @author: Tobi Delbruck, Zhe He
        @contact: tobi@ini.uzh.ch
        """

        # converting x into np.float32.
        if x.dtype is not np.float64:  # note float64 to get rounding to work
            x = x.astype(np.float64)
        f = (1 / (threshold)) * np.log(threshold)

        y = np.piecewise(
            x,
            [x <= threshold, x > threshold],
            [lambda x: x * f,
             lambda x: np.log(x)]
        )
        # important, we do a floating point round to some digits of precision
        # to avoid that adding threshold and subtracting it again results
        # in different number because first addition shoots some bits off
        # to never-never land, thus preventing the OFF events
        # that ideally follow ON events when object moves by
        y = np.around(y, 8)

        return y

    def GridGenerate(self, image_size=None, grid_mode='real'):
        '''
        :param Magnification: the magnification of the Microscope
        :param PixelSize: the PixleSize of the sCMOS or CCD
        :param EmWaveLength:  emission wavelength of sample
        :param NA:  NA(numerical aperture) of the objective
        :return:
        '''
        if image_size == None:
            y, x = self.image_size, self.image_size
        else:
            y, x = image_size, image_size

        if x % 2 == 1:
            if y % 2 == 1:
                xx, yy = torch.meshgrid(torch.arange(-(x - 1) / 2, (x + 1) / 2, 1),
                                        torch.arange(-(y - 1) / 2, (y + 1) / 2, 1))  # 空域x方向坐标为奇数，y方向坐标为奇数的情况
            else:
                xx, yy = torch.meshgrid(torch.arange(-(x - 1) / 2, (x + 1) / 2, 1),
                                        torch.arange(-y / 2, y / 2 - 1, 1))  # 空域x方向坐标为奇数，y方向坐标为偶数的情况
        else:
            if y % 2 == 1:
                xx, yy = torch.meshgrid(torch.arange(-x / 2, x / 2, 1),
                                        torch.arange(-(y - 1) / 2, (y + 1) / 2, 1))  # 空域x方向坐标为偶数，y方向坐标为奇数的情况
            else:
                xx, yy = torch.meshgrid(torch.arange(-x / 2, x / 2, 1),
                                        torch.arange(-y / 2, y / 2, 1))  # 空域x方向坐标为偶数，y方向坐标为偶数的情况
        if grid_mode == 'real':
            fx = xx * self.delta_fx
            fy = yy * self.delta_fy
            xx = xx * self.PixelSize
            yy = yy * self.PixelSize
        elif grid_mode == 'pixel':
            fx = xx * 1.0 / self.PixelSize
            fy = yy * 1.0 / self.PixelSize
        else:
            raise Exception('error grid mode')

        return xx, yy, fx, fy

    def OTF_form(self, f_grid=None, fc_ratio=1):
        if f_grid == None:
            f = self.f_grid
        else:
            f = f_grid
        f0 = self.f_cutoff * fc_ratio
        OTF = torch.where(f < f0, (2 / math.pi) * (torch.acos(f / f0) - (f / f0) * (
            pow((1 - (f / f0) ** 2), 0.5))), torch.Tensor([0]))  # Caculate the OTF support
        # OTF = torch.where(f < f0,torch.ones_like(f),torch.zeros_like(f))
        return OTF

    # def CTF_form(self,fc_ratio=1,upsample = False):
    #     f0 = fc_ratio * self.f_cutoff
    #     if upsample == False:
    #         f = self.f
    #     elif upsample == True:
    #         f = self.f_upsample
    #     CTF = torch.where(f < f0, torch.Tensor([1]), torch.Tensor([0]))
    #     return CTF

    def psf_form(self, OTF):

        OTF = OTF.squeeze()
        Numpy_OTF = OTF.numpy()
        psf = np.fft.ifftshift(np.fft.ifft2(Numpy_OTF, axes=(0, 1)), axes=(0, 1))
        psf = abs(psf)
        psf_Numpy = psf / psf.max()
        psf_tensor = torch.from_numpy(psf_Numpy)
        half_size_of_psf = int(psf.shape[0] / 2)
        half_row_of_psf = psf_tensor[half_size_of_psf]
        # a = half_row_of_psf < 1e-2
        id = torch.arange(0, half_row_of_psf.nelement())[half_row_of_psf.gt(1e-3)]
        psf_crop = psf_tensor[id[0]:id[-1] + 1, id[0]:id[-1] + 1]
        return psf_crop

    def batch_image_OTF_filter(self,batch_image):
        batch_image = batch_image.squeeze()
        ZeroPad_operation = nn.ZeroPad2d(20)
        padding_size = 20
        _, _, fx, fy = self.GridGenerate(batch_image.shape[-1] + 2 * padding_size, grid_mode='real')

        OTF_padding = self.OTF_padding.to(batch_image.device)
        batch_image_padding = ZeroPad_operation(batch_image)  # 直接进行频域OTF滤波，会有边缘信息的串扰，用zero_padding方法去除
        batch_image_padding_diffractive_spectrum = torch_2d_fftshift(
            fft.fftn(batch_image_padding, dim=[1, 2])) * OTF_padding.unsqueeze(0)
        batch_image_padding_diffractive = abs(
            fft.ifftn(torch_2d_ifftshift(batch_image_padding_diffractive_spectrum), dim=[2, 1]))
        batch_image_diffractive = batch_image_padding_diffractive[:,padding_size:-padding_size,
                                        padding_size:-padding_size]
        return batch_image_diffractive

    def padding_OTF_generate(self,padding_size = 20):
        _, _, fx, fy = self.GridGenerate(self.image_size + 2 * padding_size, grid_mode='real')
        f_grid = pow((fx ** 2 + fy ** 2), 1 / 2)  # The spatial freqneucy fr=sqrt( fx^2 + fy^2 )
        OTF_padding = self.OTF_form(f_grid)

        return OTF_padding

    def psf_form(self, OTF):

        OTF = OTF.squeeze()
        Numpy_OTF = OTF.numpy()
        psf = np.fft.ifftshift(np.fft.ifft2(Numpy_OTF, axes=(0, 1)), axes=(0, 1))
        psf = abs(psf)
        psf_Numpy = psf / psf.max()
        psf_tensor = torch.from_numpy(psf_Numpy)
        half_size_of_psf = int(psf.shape[0] / 2)
        half_row_of_psf = psf_tensor[half_size_of_psf]
        # a = half_row_of_psf < 1e-2
        id = torch.arange(0, half_row_of_psf.nelement())[half_row_of_psf.gt(1e-3)]
        psf_crop = psf_tensor[id[0]:id[-1] + 1, id[0]:id[-1] + 1]
        return psf_crop

class psf_conv_generator(nn.Module):
    def __init__(self, kernal, device):
        super(psf_conv_generator, self).__init__()
        self.kernal = kernal.to(device)

    def forward(self, HR_image):
        HR_image = HR_image.squeeze()
        kernal_size = self.kernal.size()[0]
        dim_of_HR_image = len(HR_image.size())
        if dim_of_HR_image == 4:
            min_batch = HR_image.size()[0]
            channels = HR_image.size()[1]
        elif dim_of_HR_image == 3:
            channels = HR_image.size()[0]
            HR_image = HR_image.view(1, channels, HR_image.size()[1], HR_image.size()[2])
        else:
            channels = 1
            HR_image = HR_image.view(1, 1, HR_image.size()[0], HR_image.size()[1])
        out_channel = channels
        kernel = self.kernal.expand(out_channel, 1, kernal_size, kernal_size)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False)
        return F.conv2d(HR_image, kernel.to(HR_image), stride=1, padding=int((kernal_size - 1) / 2),
                        groups=out_channel)



def generate_train_dataset(data_num=10, max_workers=None):
    simulator.output_folder = simulator.train_directory
    with tqdm(total=data_num, desc="Executing data generation", unit=" Samples") as progress_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(generate_single_STORM_DVS_data, range(data_num)):
                progress_bar.set_description("Processing %s" % result)
                progress_bar.update(1)


def generate_valid_dataset(data_num=10, max_workers=None):
    simulator.output_folder = simulator.valid_directory
    with tqdm(total=data_num, desc="Executing data generation", unit=" Samples") as progress_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(generate_single_STORM_DVS_data, range(data_num)):
                progress_bar.set_description("Processing %s" % result)
                progress_bar.update(1)


def generate_single_STORM_DVS_data(num):
    fluorophore_image_in_camera = simulator.generate_single_frame(str(num + 1))
    simulator.generate_event_data_from_single_frame(fluorophore_image_in_camera, str(num + 1))


if __name__ == '__main__':
    simulator = STORM_DVS_simulator()
    OTF = simulator.OTF
    simulator.train_directory = simulator.output_folder + '/train'
    simulator.valid_directory = simulator.output_folder + '/valid'
    generate_train_dataset(data_num=10000, max_workers=1)
    generate_valid_dataset(data_num=1000, max_workers=1)
    # for i in range(1000):
    #     generate_single_STORM_DVS_data(i)
    #     print('{}/{}'.format(i,1000))
    #TODO write a configuration file
