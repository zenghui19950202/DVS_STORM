#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：zenghui time:2020/9/24

from configparser import ConfigParser

def load_train_net_config_paras():
    config = ConfigParser()
    if config.read('../configuration.ini') != []:
        config.read('../configuration.ini')
    elif config.read('configuration.ini') != []:
        config.read('configuration.ini')
    else:
        raise Exception('directory of configuration.ini error')

    train_net_parameters = {}
    train_net_parameters['train_directory_file'] = config.get('image_file', 'SourceFileDirectory') + '/SIMdata_SR_train.txt'
    train_net_parameters['valid_directory_file'] = config.get('image_file', 'SourceFileDirectory') + '/SIMdata_SR_valid.txt'
    train_net_parameters['save_file_directory'] = config.get('image_file', 'save_file_directory')
    train_net_parameters['MAX_EVALS'] = config.getint('hyparameter', 'MAX_EVALS')  # the number of hyparameters sets
    train_net_parameters['data_generate_mode'] = config.get('data', 'data_generate_mode')
    train_net_parameters['data_input_mode'] = config.get('data', 'data_input_mode')
    train_net_parameters['net_type'] = config.get('net', 'net_type')
    train_net_parameters['LR_highway_type'] = config.get('LR_highway', 'LR_highway_type')
    train_net_parameters['data_num'] = config.getint('SIM_data_generation', 'data_num')  # the number of raw SIM images
    train_net_parameters['num_epochs'] = config.getint('hyparameter', 'num_epochs')
    train_net_parameters['image_size'] = config.getint('SIM_data_generation', 'image_size')
    train_net_parameters['opt_over'] = config.get('net', 'opt_over')


    return train_net_parameters

def load_data_generation_config_paras():
    config = ConfigParser()
    config.read('../configuration.ini')
    data_generation_parameters={}
    data_generation_parameters['output_directory'] = config.get('image_file', 'output_directory')
    data_generation_parameters['Magnification'] = config.getint('SIM_data_generation', 'Magnification')
    data_generation_parameters['PixelSizeOfCCD'] = config.getint('SIM_data_generation', 'PixelSizeOfCCD')
    data_generation_parameters['EmWaveLength'] = config.getint('SIM_data_generation', 'EmWaveLength')
    data_generation_parameters['NA'] = config.getfloat('SIM_data_generation', 'NA')
    data_generation_parameters['image_size'] = config.getint('SIM_data_generation', 'image_size')
    data_generation_parameters['fluorophore_density'] = config.getfloat('SIM_data_generation', 'fluorophore_density')
    data_generation_parameters['downsample_rate'] = config.getint('SIM_data_generation', 'downsample_rate')
    data_generation_parameters['parallel_frames'] = config.getint('SIM_data_generation', 'parallel_frames')


    return data_generation_parameters

if __name__ == '__main__':
    a = load_data_generation_config_paras()