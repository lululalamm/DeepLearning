
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat

from .model_zoo.get_models import get_detection_model,get_landmark_model, get_ageGender_model,get_recognition_model
from .face.get_result import *
from .utils.util_common import draw_result,draw_result_sim
from .data.image import read_image,read_image_retinaTorch

import time


def get_preds(pred_dir,detection_model,thresh,input_size,save_dir,target_size,max_size,multiscale,resize_way,write_path):
    events = [ name for name in os.listdir(pred_dir) if not ".DS" in name ]
    boxes = dict()
    pbar = tqdm.tqdm(events)
    all_times=0

    os.makedirs(save_dir, exist_ok=True)

    if write_path:
        time_return=True
        time_sum = {'rescale': 0,
                    'net_out': 0,
                    'forward': 0,
                    'post1': 0,
                    'get_detect': 0,
                    'post2': 0,
                    'max_det': 0,
                    'total':0}
        box_count=0
    else:
        time_return=False
    ct=-1
    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        if not os.path.exists(os.path.join(save_dir,event)):
            os.mkdir(os.path.join(save_dir,event))
        event_images = [ name for name in os.listdir(event_dir) if not ".DS" in name]
        current_event = dict()
        sub_times=0
        for imgname in event_images:
            ct+=1
            img_path = os.path.join(event_dir,imgname)
            file_name = imgname.split(".")[0]

            txt_path = os.path.join(save_dir,event,file_name+".txt")


            with open(txt_path,'w') as f:
                f.writelines(file_name+"\n")
                
                
                img = read_image(img_path)
                #img = read_image_retinaTorch(img_path)

                if not multiscale:
                    start = time.time()
                    faces, time_dict = get_detection(detection_name,detection_model,img,thresh=thresh,input_size=input_size,resize_way=resize_way,time_return=time_return)
                    end = time.time()
                else:
                    start = time.time()
                    faces, time_dict = get_detection(detection_name,detection_model,img,thresh=thresh,target_size=target_size,max_size=max_size,resize_way=resize_way,time_return=time_return)
                    end = time.time()

                box_count+=len(faces)
                f.writelines(str(len(faces))+"\n")

                for face in faces:
                    x1 = face.bbox[0]
                    y1 = face.bbox[1]
                    w = face.bbox[2]-face.bbox[0]
                    h = face.bbox[3]-face.bbox[1]

                    line = "{} {} {} {} {} \n".format(x1,y1,w,h,face.score)

                    f.writelines(line)

                if write_path:
                    if ct==0: continue

                    time_dict['total']=(end-start)*1000

                    for key in time_dict.keys():
                        time_sum[key]+=time_dict[key]


    if write_path:
        with open(write_path,'a') as time_f:
            if detection_model.model_type=='onnx':
                model_info = "{}_{}-{}_".format(detection_model.model_name,detection_model.model_type,detection_model.net.device)
            else:
                model_info = "{}_{}_".format(detection_model.model_name,detection_model.model_type)
            if multiscale:
                model_info += "multi_target-{}_max-{}\n".format(target_size,max_size)
            else:
                model_info += "single_input-{}\n".format(input_size)

            time_f.writelines(model_info)

            time_f.writelines("total Boxes: {}\n".format(box_count))
            for key in time_sum.keys():
                line = "\t{}: {} ms".format(key,(time_sum[key]/(ct)))
                time_f.writelines(line+"\n")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="WiderFace/WIDER_val/images/")
    parser.add_argument('--dt_name',type=str,default='scrfd')
    parser.add_argument('--dt_path',type=str,default='scrfd_10g_bnkps.v8.trt')
    parser.add_argument('--thresh',type=float,default=0.02)
    parser.add_argument('--save_format',type=str,default="./result_txt/result_WF_{}/")
    parser.add_argument('--input_size',nargs='+', type=int, default=[640,640],help="input size height, width")
    parser.add_argument('--target_size', type=int, default=0,help="if multi scale")
    parser.add_argument('--max_size', type=int, default=0,help="if multi scale")
    parser.add_argument('--load_multi',action='store_true',help='if load model multi TRT')
    parser.add_argument('--multiscale',action='store_true',help='if use multiscale image (onnx, trt)')
    parser.add_argument('--onnx_cpu',action='store_true',help="if inference onnx in cpu")
    parser.add_argument('--resize_way',type=str,default='resize',help='for retinaface torch,,, resize or pad')
    parser.add_argument('--write_time',action='store_true',help='if write detection time .. in save txt')

    args = parser.parse_args()

    #if not os.path.exists("./result_txt/"):
    #    os.mkdir("./result_txt/")

    detection_name= args.dt_name
    detection_path = args.dt_path
    detection_thresh = args.thresh
    detection_size = tuple(args.input_size)

    if args.onnx_cpu:
        onnx_device="cpu"
    else:
        onnx_device="cuda"

    save_result = args.save_format.format(detection_path.split("/")[-1])

    if ".vino" in detection_path:
        dname = detection_path.split(".vino")[0]
        detection_path = [dname+".xml",dname+".bin"]

    if args.write_time:
        write_path = os.path.join(*args.save_format.split("/")[:-1],"result_time.txt")
    else:
        write_path=""

    print(detection_name,detection_path)
    detection_model = get_detection_model(detection_name,detection_path,load_multi=args.load_multi,onnx_device=onnx_device)
    get_preds(args.pred, detection_model,detection_thresh,detection_size,save_result,args.target_size, args.max_size,args.multiscale,args.resize_way,write_path)
    print("Save",save_result)
