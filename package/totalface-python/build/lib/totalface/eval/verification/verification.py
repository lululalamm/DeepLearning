import argparse

import cv2
import os, sys
import numpy as np
import torch
from collections import OrderedDict
import tensorrt as trt

from .eval_util.verification import load_bin,test,test_convert
from ...model_zoo.get_models import get_recognition_model


def get_args():
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data_dir', 
                         default='/data/notebook/NAS/FaceRecognition/datasets/ms1m-retinaface-t1/',
                         help='')
    parser.add_argument('--model_name',
                         default='',
                         help='model name: "arcface"')
    parser.add_argument('--model_path',
                         default='',
                         help='model path')                     
    parser.add_argument('--network',
                         default='r50',
                         help='network name: "r50 r100 r200 ..."')
    parser.add_argument('--out_size', type=int,default=112, help='aligned face image output size')
    parser.add_argument('--num_features', type=int,default=512, help='embedding feature size')
    parser.add_argument('--fp16', action='store_true', default=False,help='if fp16,,')

    parser.add_argument('--onnx_device',type=str,default='cuda')

    parser.add_argument('--targets',nargs='+',type=str, \
                        default=['lfw_mask', 'cfp_fp_mask', 'agedb_30_mask', 'lfw', 'cfp_fp', 'agedb_30'])
    parser.add_argument('--save_base',type=str,default="/data/",help="save txt base path")

    args = parser.parse_args()

    return args


def verification_run(data_dir,model_name,model_path,network,out_size,num_features,fp16,onnx_device,targets,save_base):
   
    ds_list = {}
    for target in targets:
        dataset = load_bin(os.path.join(data_dir, target+".bin"),(112,112))
        ds_list[target] = dataset

    try:
        assert model_name in ['arcface']
    except:
        print("Available model names: arcface")
        exit()

    model_type = model_path.split(".")[-1]
    save_path = os.path.join(save_base,"result_{}-{}_{}.txt".format(model_name,network,model_type))

    if not os.path.exists(save_base):
        os.makedirs(save_base)

    if ".vino" in model_path:
        mname = model_path.split(".vino")[0]
        model_path = [mname+".xml",mname+".bin"]

    if model_type=='trt':
        img_cuda=True
    else:
        img_cuda=False

    if not network=='mbf':
        network_name = network if 'm' not in network else network.replace('m','')
        network_name = network_name if '-pfc' not in network_name else network_name.replace('-pfc','')
    else:
        network_name=network

    model = get_recognition_model(model_name,model_path, \
                                  out_size=out_size, num_features=num_features, \
                                  network=network_name, fp16=fp16, load_multi=True, \
                                  input_mean=0.0, input_std=1.0,not_norm=True,torch_image=True,transform=False, \
                                  onnx_device=onnx_device
                                  )

    # load model
    

    for target in targets:
        dataset = ds_list[target]

        acc1, std1, acc2, std2, xnorm, embeddings_list = test_convert(dataset,img_cuda, model, 10, 10)

        print("Test {} - {}".format(network, target))
        print('[%s]XNorm: %f' % (target, xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (target, acc2, std2))
        print()
        
        with open(save_path, 'a') as f:
            f.write("[%s]XNorm: %f\n" % (target, xnorm))
            f.write("[%s]Accuracy-Flip: %1.5f+-%1.5f\n\n" % (target, acc2, std2))


if __name__ == '__main__':
    args = get_args()

    data_dir = args.data_dir
    targets = args.targets

    ds_list = {}
    for target in targets:
        dataset = load_bin(os.path.join(data_dir, target+".bin"),(112,112))
        ds_list[target] = dataset


    model_name = args.model_name
    try:
        assert model_name in ['arcface']
    except:
        print("Available model names: arcface")
        exit()
    model_path = args.model_path
    model_type = model_path.split(".")[-1]
    save_path = os.path.join(args.save_base,"result_{}-{}_{}.txt".format(model_name,args.network,model_type))

    if not os.path.exists(args.save_base):
        os.makedirs(args.save_base)

    if ".vino" in model_path:
        mname = model_path.split(".vino")[0]
        model_path = [mname+".xml",mname+".bin"]

    model = get_recognition_model(model_name,model_path, \
                                  out_size=args.out_size, num_features=args.num_features, \
                                  network=args.network, fp16=args.fp16, load_multi=True, \
                                  input_mean=0.0, input_std=1.0,not_norm=True,torch_image=True,transform=False, \
                                  onnx_device=args.onnx_device
                                  )

    # load model
    

    for target in targets:
        dataset = ds_list[target]

        if model_type in ['pt','pth']:
            acc1, std1, acc2, std2, xnorm, embeddings_list = test(dataset, model, 10, 10)
        else:
            acc1, std1, acc2, std2, xnorm, embeddings_list = test_convert(dataset, model, 10, 10)

        print("Test {} - {}".format(args.network, target))
        print('[%s]XNorm: %f' % (target, xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (target, acc2, std2))
        print()
        
        with open(save_path, 'a') as f:
            f.write("[%s]XNorm: %f\n" % (target, xnorm))
            f.write("[%s]Accuracy-Flip: %1.5f+-%1.5f\n\n" % (target, acc2, std2))

