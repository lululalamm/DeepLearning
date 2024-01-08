# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training
"""

import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel

# new_dict={'org_1_80x60':'org_1_80x60',
#                 '1_80x80':'1_80x80',
#                 '2.7_80x80':'2.7_80x80',
#                 '4_80x80':'4_80x80'}

def get_default_config():
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 1e-1
    # [9, 13, 15]
    conf.milestones = [10, 15, 22]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 25
    conf.momentum = 0.9
    conf.batch_size = 1024

    # model
    conf.num_classes = 2
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.train_root_path = './'
    conf.train_root_txt_path = 'temp.txt'
    #conf.train_root_h5 = "FaceInTheWild/unmask/Training/{}.h5"

    # save file path
    conf.snapshot_dir_path = './saved_logs/snapshot_ycrcb'

    # log path
    conf.log_path = './saved_logs/jobs_ycrcb'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 5000

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name

    
    return conf

def update_config_txt(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name

    conf.new_base = args.new_format.format(args.patch_info)

    return conf

def update_config_h5(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    conf.train_root_h5 = args.h5_format
    
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    conf.network = args.network
    conf.resnet_name = args.resnet

    if conf.network=='resnet':
        if args.norm_input:
            job_name = 'Anti_Spoofing_{}_resnet_{}_{}_norm'.format(args.patch_info,conf.resnet_name,args.data_name)
        else:
            job_name = 'Anti_Spoofing_{}_resnet_{}_{}'.format(args.patch_info,conf.resnet_name,args.data_name)
        conf.ft_height = 20
        conf.ft_width = 20
    else:
        conf.ft_height = 2*conf.kernel_size[0]
        conf.ft_width = 2*conf.kernel_size[1]
        if args.norm_input:
            job_name = 'Anti_Spoofing_{}_{}_norm'.format(args.patch_info,args.data_name)
        else:
            job_name = 'Anti_Spoofing_{}_{}'.format(args.patch_info,args.data_name)
    #job_name = 'Test_flag_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name

    conf.h5_path = conf.train_root_h5.format(args.patch_info)
    if "wild" in conf.h5_path or "Wild" in conf.h5_path:
        conf.save_every = 5000
    else:
        conf.save_every = 800
        conf.lr = 0.01

    

    return conf
