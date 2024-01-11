import argparse

import cv2
import os, sys
import numpy as np

from ...model_zoo.get_models import get_recognition_model
from .gen_features_lst import gen_features_run
from .remove_noises import remove_noises_main as remove_noises
from .run_experiment import run_experiment_main as run_experiment
from .megaface_results import get_result


# gen features 
def megaface_gen_features(model_name, model_path,scrub_dir,scrub_lst,mega_dir,mega_lst,save_base, \
                            network_ori,out_size=112,num_features=512,fp16=True,onnx_device='cuda',batch_size=128):

    try:
        assert model_name in ['arcface']
    except:
        print("Available model names: arcface")
        exit()

    model_type = model_path.split(".")[-1]

    if ".vino" in model_path:
        mname = model_path.split(".vino")[0]
        model_path = [mname+".xml",mname+".bin"]

    if model_type=='trt':
        img_cuda=True
    else:
        img_cuda=False

    if model_type=='onnx':
        model_type='onnx-{}'.format(onnx_device)

    if not network_ori=='mbf':
        network_name = network_ori if 'm' not in network_ori else network_ori.replace('m','')
        network_name = network_name if '-pfc' not in network_name else network_name.replace('-pfc','')
    else:
        network_name='mbf'

    print("Load path: {}".format(model_path))
    print("Load type: {}".format(model_type))
    print("Load multi: True")

    # load model
    model = get_recognition_model(model_name,model_path, \
                                  out_size=out_size, num_features=num_features, \
                                  network=network_name, fp16=fp16, load_multi=True, \
                                  input_mean=0.0, input_std=1.0,not_norm=True,torch_image=True,transform=False, \
                                  onnx_device=onnx_device
                                  )
    
    save_base = '{}-{}_{}'.format(save_base,model_type,network_ori)

    gen_features_run(model,network_ori,scrub_dir,scrub_lst,mega_dir,mega_lst,save_base,img_cuda,batch_size)

    return model_type

# remove_noises
def megaface_remove_noises(facescrub_noises,megaface_noises,algo,facescrub_lst,megaface_lst,feature_dir_input, feature_dir_output,model_type):
    
    remove_noises(facescrub_noises,megaface_noises,algo,facescrub_lst,megaface_lst,feature_dir_input, feature_dir_output,model_type)

# run experiments
def megaface_run_experiments(jb_model,exe_id,exe_fuse,magaface_lst_json,prob_lst_json, \
                            features_mega,features_prob,file_ending,save_base,sizes,num_sets,dm):

    run_experiment(jb_model,exe_id,exe_fuse,magaface_lst_json,prob_lst_json,features_mega,features_prob,file_ending,save_base,sizes,num_sets,dm)


# get results
def megaface_results(exp_base,save_dir,netname='r',dist_list=[]):

    get_result(exp_base,save_dir,netname=netname,dist_list=dist_list)




