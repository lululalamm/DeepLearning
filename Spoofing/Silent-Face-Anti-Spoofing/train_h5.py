# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:39
# @Author : zhuying
# @Company : Minivision
# @File : train.py
# @Software : PyCharm

import argparse
import os
from src.train_main import TrainMain
from src.train_main_norm import TrainMain as TrainMain_norm
# from src.default_config import get_default_config, update_config
from src.aihub_config import get_default_config, update_config_txt,update_config_h5

import os
import time
from datetime import datetime
import json


def parse_args():
    """parsing and configuration"""
    desc = "Silence-FAS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_ids", type=str, default="01", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="2.7_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    parser.add_argument("--network",type=str,default='nasnet')
    parser.add_argument("--resnet",type=str,default='r100')
    parser.add_argument('--data_name',type=str,default='celebA')
    parser.add_argument('--h5_format',type=str,default="")
    parser.add_argument('--norm_input',action='store_true',default=False)
    parser.add_argument('--ycrcb',action='store_true',default=False)
    args = parser.parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    print("cuda devices:",cuda_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = [x for x in range(len(cuda_devices))]

    return args


if __name__ == "__main__":

    slack_api = None
    args = parse_args()

    print("Start")
    conf = get_default_config()
    conf = update_config_h5(args, conf)

    print("update config h5")
    if args.norm_input:
        trainer = TrainMain_norm(conf,h5=True,ray=False,slack_api=slack_api,ycrcb=args.ycrcb)
    else:
        trainer = TrainMain(conf,h5=True,ray=False,slack_api=slack_api,ycrcb=args.ycrcb)
    print("trainer init")
    trainer.train_model()

