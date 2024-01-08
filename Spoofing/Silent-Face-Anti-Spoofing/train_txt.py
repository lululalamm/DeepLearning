# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:39
# @Author : zhuying
# @Company : Minivision
# @File : train.py
# @Software : PyCharm

import argparse
import os
from src.train_main import TrainMain
# from src.default_config import get_default_config, update_config
from src.aihub_config import get_default_config, update_config_txt


def parse_args():
    """parsing and configuration"""
    desc = "Silence-FAS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_ids", type=str, default="01", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="2.7_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    parser.add_argument("--new_format",type=str,default="/data/shared/for_anti-spoofing/unmask/Training/{}/")
    args = parser.parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = [x for x in range(len(cuda_devices))]

    return args


if __name__ == "__main__":
                
    args = parse_args()
    print("Start")
    conf = get_default_config()
    conf = update_config_txt(args, conf)

    print("update config txt")
    trainer = TrainMain(conf,txt=True,ray=True)
    print("trainer init")
    trainer.train_model()

