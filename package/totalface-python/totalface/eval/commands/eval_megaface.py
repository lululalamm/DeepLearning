from argparse import ArgumentParser

from . import BaseInsightFaceCLICommand
import cv2
import os, sys
import numpy as np
import torch
from collections import OrderedDict

from ..megaface.megaface_run import megaface_gen_features,megaface_remove_noises,megaface_run_experiments,megaface_results


def evaluation_megaface(args):
    return EvalMegaface(args.data_dir, args.model_name, args.model_path, args.network,\
                        args.out_size, args.num_features, args.fp16, args.onnx_device, args.sizes, args.num_sets, \
                        args.delete_matrices, args.batch_size, args.save_base,args.feature_pass,args.remove_pass,args.exp_pass,args.model_type)

class EvalMegaface(BaseInsightFaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        parser = parser.add_parser("megaface")
        parser.add_argument('--data_dir', 
                         default='MegaFace/megaface/',
                         help='data directory path : data_dir/data/~ data_dir/devkit/~')

        # model args
        parser.add_argument('--model_name',
                            default='arcface',
                            choices=['arcface'],
                            help='model name: "arcface"')
        parser.add_argument('--model_path',
                            default='',
                            help='model full path')                     
        parser.add_argument('--network',
                            default='r50',
                            choices=['r18','r34','r50','r50m','r100','r100m','r100m-pfc','r200','r200m','r200-pfc','r200m-pfc','mbf','eb0','ev2_s'],
                            help='network names \n - glint360k : r50 r100 r100m r100m-pfc r200 r200-pfc r200m-pfc \n ms1m :r18, r34 ,r50 ,r100, r50m r100m mbf')

        parser.add_argument('--onnx_device',type=str,default='cuda',choices=['cpu','cuda'],help='onnx inference env : cpu or cuda')
        parser.add_argument('--out_size', type=int,default=112, help='aligned face image output size (default 112)')
        parser.add_argument('--num_features', type=int,default=512, help='embedding feature size (default 512)')
        parser.add_argument('--fp16', action='store_true', help='load fp16 model, default False')

        # infer args
        parser.add_argument('--sizes',nargs='+',type=int, \
                            default=[10,100,1000,10000,100000,1000000], \
                             help='test sizes : 10 100 1000 10000 100000 1000000 (default all)')
        parser.add_argument('--num_sets',type=int,default=1,help='default 1')
        parser.add_argument('--delete_matrices',action='store_true',default=False,help='default False')
        parser.add_argument('--batch_size',type=int,default=128,help='default 128')

        parser.add_argument('--feature_pass',action='store_true',default=False)
        parser.add_argument('--remove_pass',action='store_true',default=False)
        parser.add_argument('--exp_pass',action='store_true',default=False)
        parser.add_argument('--model_type',type=str,default="",help="if set model_type")

        parser.add_argument('--save_base',type=str,default="result_megaface/",help="save result dir (default result_megaface/)")
        parser.set_defaults(func=evaluation_megaface)

    def __init__(self, data_dir,model_name,model_path,network_ori,out_size,num_features,fp16,onnx_device,sizes,num_sets,delete_matrices,batch_size,save_base, \
                feature_pass,remove_pass,exp_pass,model_type):
        self.data_dir = data_dir
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        
        self.network = network_ori
        
        self.out_size = out_size
        self.num_features = num_features
        self.fp16 = fp16
        self.onnx_device = onnx_device

        if self.network=='mbf':
            self.netname='mbf'
        else:
            self.netname = self.network[0]

        self.save_base = save_base
        self.feature_input = os.path.join(self.save_base,"save_features")
        self.feature_output = os.path.join(self.save_base,"save_features_out")

        self.scrub_dir = os.path.join(self.data_dir,'data/megaface/facescrub_images')
        self.scrub_lst = os.path.join(self.data_dir,'data/facescrub_lst')
        self.mega_dir = os.path.join(self.data_dir,'data/megaface/megaface_images')
        self.mega_lst = os.path.join(self.data_dir,'data/megaface_lst')

        self.facescrub_noises = os.path.join(self.data_dir,'data/facescrub_noises.txt')
        self.megaface_noises = os.path.join(self.data_dir,'data/megaface_noises.txt')

        self.jb_model = os.path.join(self.data_dir,'devkit/models/jb_identity.bin')
        self.exe_id = os.path.join(self.data_dir,'devkit/bin/Identification')
        self.exe_fuse = os.path.join(self.data_dir,'devkit/bin/FuseResults')
        self.magaface_lst_json = os.path.join(self.data_dir,'devkit/templatelists/megaface_features_list.json')
        self.prob_lst_json = os.path.join(self.data_dir,'devkit/templatelists/facescrub_features_list.json')

        self.file_ending = '_{}.bin'.format(network_ori)

        self.save_exp = os.path.join(self.save_base,"mega_experiments")
        self.save_result = os.path.join(self.save_base,"mega_result")

        self.sizes = sizes
        self.num_sets = num_sets
        self.delete_matrices = delete_matrices

        self.feature_pass = feature_pass
        self.remove_pass = remove_pass
        self.exp_pass = exp_pass
        self.model_type=model_type

    def run(self):

        if self.feature_pass:
            print("Gen feature Pass...")
            if self.model_path:
                self.model_type = self.model_path.split(".")[-1]

                if self.model_type=='onnx':
                    self.model_type='onnx-{}'.format(self.onnx_device)
                    print("onnx device:",self.onnx_device)

        else:
            print("Step 1. gen features")
            self.model_type = megaface_gen_features(self.model_name, self.model_path, \
                                                    self.scrub_dir,self.scrub_lst,self.mega_dir,self.mega_lst, \
                                                    self.feature_input ,self.network,out_size=self.out_size, num_features=self.num_features, \
                                                    fp16=self.fp16, onnx_device=self.onnx_device,batch_size=self.batch_size)
        print("model type:",self.model_type)

        if self.remove_pass:
            print("Remove noises Pass...")
        else:
            print("Step 2. remove noises")
            megaface_remove_noises(self.facescrub_noises,self.megaface_noises,self.network, \
                                    self.scrub_lst,self.mega_lst,self.feature_input, self.feature_output,self.model_type)

        if self.exp_pass:
            print("Run experiments Pass...")
        else:
            print("Step 3. run experiments")
            self.features_mega = self.feature_output+"-{}_{}/megaface".format(self.model_type, self.network)
            self.features_prob = self.feature_output+"-{}_{}/facescrub".format(self.model_type, self.network)
            megaface_run_experiments(self.jb_model,self.exe_id,self.exe_fuse,self.magaface_lst_json,self.prob_lst_json, \
                                    self.features_mega,self.features_prob,self.file_ending,self.save_exp,self.sizes,self.num_sets,self.delete_matrices)

        print("Step 4. final results")
        megaface_results(self.save_exp,self.save_result,netname=self.netname,dist_list=self.sizes)