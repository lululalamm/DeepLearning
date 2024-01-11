from argparse import ArgumentParser

from . import BaseInsightFaceCLICommand
import cv2
import os, sys
import numpy as np
import torch
from collections import OrderedDict

from ..ijbc.ijbc_run import ijbc_run

def evaluation_ijbc(args):
    return EvalIjbc(args.data_dir, args.model_name, args.model_path, args.network,\
                             args.out_size, args.num_features, args.fp16, args.onnx_device, \
                             args.not_use_norm_score, args.not_use_detector_score, args.not_use_flip_test, \
                             args.gpuid, args.job, args.batch_size, args.target, args.save_base)

class EvalIjbc(BaseInsightFaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        parser = parser.add_parser("ijbc")
        parser.add_argument('--data_dir', 
                         default='IJB/IJB_release/IJBC/',
                         help='data directory path : data_dir/meta data_dir/loose_crop ...')

        # model args
        parser.add_argument('--model_name',
                            default='arcface',
                            choices=['arcface'],
                            help='model names : "arcface"')
        parser.add_argument('--model_path',
                            default='glint360k-r50-arcface_multiple.onnx',
                            help='model full path')                     
        parser.add_argument('--network',
                            default='r50',
                            choices=['r18','r34','r50','r50m','r100','r100m','r100m-pfc','r200','r200m','r200-pfc','r200m-pfc','mbf','eb0','ev2_s'],
                            help='network names \n - glint360k : r50 r100 r100m r100m-pfc r200 r200-pfc r200m-pfc \n ms1m : r18 r34 r50 r100 r50m r100m mbf')
        parser.add_argument('--out_size', type=int,default=112, help='aligned face image output size (default 112)')
        parser.add_argument('--num_features', type=int,default=512, help='embedding feature size (default 512)')
        parser.add_argument('--fp16', action='store_true', help='load fp16 model, default False')
        parser.add_argument('--onnx_device',type=str,default='cuda',choices=['cpu','cuda'],help='onnx inference env : cpu or cuda')

        # infer args
        parser.add_argument('--not_use_norm_score',action='store_false',default=True,help='if not use norm score (default use)')
        parser.add_argument('--not_use_detector_score',action='store_false',default=True,help='if not use detector score (default use)')
        parser.add_argument('--not_use_flip_test',action='store_false',default=True,help='if not use flip test (default use)')

        parser.add_argument('--gpuid',type=int,default=None,help='need set gpuid (default None)')
        parser.add_argument('--job',type=str,default='insightface',help='job name (default insightface)')
        parser.add_argument('--batch_size',type=int,default=128,help='default 128')
        parser.add_argument('--target',type=str,default='IJBC',choices=['IJBC','IJBB'],help='target name (default IJBC)')

        parser.add_argument('--save_base',type=str,default="result_ijbc/",help="save result dir (default result_ijbc/)")
        parser.set_defaults(func=evaluation_ijbc)

    def __init__(self, data_dir,model_name,model_path,network,out_size,num_features,fp16,onnx_device, \
                not_use_norm_score,not_use_detector_score,not_use_flip_test, \
                gpuid,job,batch_size,target,save_base):
        self.data_dir = data_dir
        self.model_name = model_name
        self.model_path = model_path
        
        self.network = network
        
        self.out_size = out_size
        self.num_features = num_features
        self.fp16 = fp16
        self.onnx_device = onnx_device

        self.not_use_norm_score = not_use_norm_score
        self.not_use_detector_score = not_use_detector_score
        self.not_use_flip_test = not_use_flip_test

        self.gpuid = gpuid
        self.job = job
        self.batch_size = batch_size

        self.target = target
        self.save_base = save_base

    def run(self):
        ijbc_run(self.model_name, self.model_path,self.network,self.data_dir,self.save_base, \
        out_size=self.out_size,num_features=self.num_features,fp16=self.fp16,onnx_device=self.onnx_device, \
        use_norm_score=self.not_use_norm_score,use_detector_score=self.not_use_detector_score,use_flip_test=self.not_use_flip_test \
        ,gpuid=self.gpuid,job=self.job,batch_size=self.batch_size,target=self.target)