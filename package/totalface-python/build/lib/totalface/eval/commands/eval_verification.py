from argparse import ArgumentParser

from . import BaseInsightFaceCLICommand
import cv2
import os, sys
import numpy as np
import torch
from collections import OrderedDict

from ..verification.verification import verification_run

def evaluation_verification(args):
    return EvalVerifiacation(args.data_dir, args.model_name, args.model_path, args.network,\
                             args.out_size, args.num_features, args.fp16, args.onnx_device, \
                             args.targets, args.save_base)

class EvalVerifiacation(BaseInsightFaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        parser = parser.add_parser("verification")
        parser.add_argument('--data_dir', 
                         default='/data/notebook/shared/Face/FaceRecognition/datasets/eval/bins/',
                         help='data directory path : data_dir/~.bin')
        parser.add_argument('--model_name',
                            default='arcface',
                            help='model names : "arcface"')
        parser.add_argument('--model_path',
                            default='/data/notebook/NAS/PTAS_Shared/resource/model/face/embedding/glint360k-r50-arcface_multiple.onnx',
                            help='model full path')                     
        parser.add_argument('--network',
                            default='r50',
                            choices=['r18','r34','r50','r50m','r100','r100m','r100m-pfc','r200','r200m','r200-pfc','r200m-pfc','mbf','eb0','ev2_s'],
                            help='network names \n - glint360k : r50 r100 r100m r100m-pfc r200 r200-pfc r200m-pfc \n ms1m :r18, r34 ,r50 , r100, r50m r100m mbf')
        parser.add_argument('--out_size', type=int,default=112, help='aligned face image output size (default 112)')
        parser.add_argument('--num_features', type=int,default=512, help='embedding feature size (default 512)')
        parser.add_argument('--fp16', action='store_true', help='load fp16 model, default False')

        parser.add_argument('--onnx_device',type=str,default='cuda',choices=['cpu','cuda'],help='onnx inference env : cpu or cuda')

        parser.add_argument('--targets',nargs='+',type=str, \
                            default=['lfw_mask', 'cfp_fp_mask', 'agedb_30_mask', 'lfw', 'cfp_fp', 'agedb_30'],
                            help='target bin names : lfw_mask cfp_fp_mask agedb_30_mask lfw cfp_fp agedb_30')
        parser.add_argument('--save_base',type=str,default="/data/result_verification/",help="save result dir (default /data/result_verification/ )")
        parser.set_defaults(func=evaluation_verification)

    def __init__(self, data_dir,model_name,model_path,network,out_size,num_features,fp16,onnx_device,targets,save_base):
        self.data_dir = data_dir
        self.model_name = model_name
        self.model_path = model_path

        self.network = network
        
        self.out_size = out_size
        self.num_features = num_features
        self.fp16 = fp16
        self.onnx_device = onnx_device
        self.targets = targets
        self.save_base = save_base

    def run(self):
        verification_run( data_dir=self.data_dir ,model_name=self.model_name ,model_path=self.model_path, \
                            network=self.network,out_size=self.out_size,num_features=self.num_features,fp16=self.fp16, \
                            onnx_device=self.onnx_device,targets=self.targets,save_base=self.save_base)