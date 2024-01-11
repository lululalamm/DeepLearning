import argparse

import cv2
import os, sys
import numpy as np

from .eval_ijbc import load_model,run_ijbc
from ...model_zoo.get_models import get_recognition_model


def ijbc_run(model_name, model_path,network,image_path,result_dir, \
        out_size=112,num_features=512,fp16=True,onnx_device='cuda', \
        use_norm_score=True,use_detector_score=True,use_flip_test=True,gpuid=None,job='insightface',batch_size=128,target='IJBC'):
                                
    model, model_type,img_cuda = load_model(model_name,model_path,network,out_size=out_size,num_features=num_features,fp16=fp16,onnx_device=onnx_device)
    print("Load path: {}".format(model_path))
    print("Load type: {}".format(model_type))
    print("Load multi: True")

    print("start test ijbc")
    run_ijbc(image_path,model,result_dir,network,img_cuda,use_norm_score=use_norm_score, \
            use_detector_score=use_detector_score,use_flip_test=use_flip_test,gpuid=gpuid,job=job,batch_size=batch_size,target=target)


