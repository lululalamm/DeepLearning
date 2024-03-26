import argparse

import cv2
import numpy as np
import torch
import os
import pickle
import time

from backbones import get_cmt_model_c14
import random
import h5py
import torch.nn as nn

from sklearn.metrics import mean_absolute_error
from collections import OrderedDict

from torchvision import transforms as T


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_image(img,convert):

    img = cv2.resize(img, (112, 112))

    if convert:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return img

def get_pred(sf_gender, sf_age):#,resize):

    age_dict=['-10s','20s','30s','40s','50s','60s','70-']

    age_female = sf_age[:,:7]
    age_male = sf_age[:,7:]

    p_female = sf_gender[:,0][0]*age_female
    p_male = sf_gender[:,1][0]*age_male
    p_age = p_female+p_male

    pred_gender = np.argmax(sf_gender)
    pred_age_idx = np.argmax(p_age)
    pred_age = age_dict[pred_age_idx]

    return pred_gender, pred_age, pred_age_idx, p_age


@torch.no_grad()
def test(args):


    weight, name, path, backbone_name, embedding_size  = args.weight, args.network, args.h5_path, args.backbone_name, args.embedding_size
    db_name = args.db
    
    net = get_cmt_model_c14("", embedding_size)

    load_weight = torch.load(weight)
    if type(load_weight)==OrderedDict:
        try:
            net.load_state_dict(load_weight)
        except:
            new_state_dict = OrderedDict()
            for n, v in load_weight.items():
                name = n.replace("module.","") 
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
    else:
        try:
            net.load_state_dict(load_weight.module.state_dict())
        except:
            net.load_state_dict(load_weight.state_dict())
 

    net.to(device)
    net.eval()

    save_base = "/data/notebook/yoonms/face_agender/save_pkl/c14/"
    weight_name = weight.split("/")[-1]
    save_base = os.path.join(save_base,weight_name)
    if not os.path.exists(save_base):
        os.mkdir(save_base)
    save_pkl = os.path.join(save_base,db_name+".pkl")

    if args.save_txt:
        save_base = "./save_txt_result_220609_c14/"
        weight_name = weight.split("/")[-1]

        if not os.path.exists(save_base):
            os.mkdir(save_base)
        save_base = os.path.join(save_base,weight_name)
        if not os.path.exists(save_base):
            os.mkdir(save_base)
        
        test_name = args.db
        save_txt_path = os.path.join(save_base,"{}.txt".format(test_name))
        txt_file = open(save_txt_path,'a')
        txt_file.writelines("idx g_age g_gender p_age p_gender p_age_name\n")

    all_time=[]

    d = h5py.File(path,'r')

    images = d["image"]
    if "appa" in path:
        ages = d['age_appa']
    else:   
        ages = d['age']
    genders = d["gender"]

    total = len(ages)

    with torch.no_grad():
        sm_list=[]
        for li in range(len(ages)):
            sm_lb=[]
            if li%1000==0: print(li)

            image = images[li]

            age = int(float(ages[li]))
            gender = int(float(genders[li]))

            img = read_image(image,args.convert)

            img = img.to(device)
            start = time.time()

            sf_gender, sf_age = net(img)

            sf_gender = sf_gender.cpu().numpy()
            sf_age = sf_age.cpu().numpy()

            pred_g,pred_a,pred_a_idx, p_age = get_pred(sf_gender,sf_age)#,img.size(0))

            sm_lb.append(p_age)
            sm_lb.append(sf_gender)
            sm_lb.append(age)
            sm_lb.append(gender)
            sm_list.append(sm_lb)

            if args.save_txt:
                new_line = "{} {} {} {} {} {}\n".format(li,age,gender,pred_a_idx,pred_g,pred_a)
                txt_file.writelines(new_line)

            end = time.time()

            infer_time = (end-start)
            all_time.append(infer_time)

    pickle.dump(sm_list,open(save_pkl,'wb'))
    print("Save",save_pkl)

    print("Inference time (total time/ total len) (s):",sum(all_time)/total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='ms1mv3_new_agengender_211110_nf_aligned/best/epoch_4_step_8295_val_loss_0.6511_val_acc_0.7838_best.pth')
    parser.add_argument('--h5_path', type=str, default="/data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112/test_wiki_dlib_aligned.h5")
    parser.add_argument('--backbone_name',type=str,default='res50',help='res50, res100, mbf')
    parser.add_argument("--embedding_size",type=int,default=512)
    parser.add_argument("--convert",type=lambda x: (str(x).lower() == 'true'),default=False)
    parser.add_argument("--save_txt",type=lambda x: (str(x).lower() == 'true'),default=False)
    parser.add_argument("--db",type=str,default=None)


    args = parser.parse_args()
    test(args)

  # dlib 112x112 margin 40 
# /data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112/test_wiki_dlib_aligned.h5
# /data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112/test_japan_dlib_aligned.h5
# /data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112/test_celeb_dlib_aligned.h5

# dlib 112x112 nomargin
# /data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112_nomargin/kceleb_dlib_aligned.h5
# /data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112_nomargin/jceleb_dlib_aligned.h5
# /data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112_nomargin/nceleb_dlib_aligned.h5

# mtcnn 112x112 nomargin
# /data/notebook/NAS/Gender-Age/test_data/mtcnn_aligned_112x112/arcface_wiki_korean_finish_new.h5
# /data/notebook/NAS/Gender-Age/test_data/mtcnn_aligned_112x112/arcface_japan_celeb_new.h5
# /data/notebook/NAS/Gender-Age/test_data/mtcnn_aligned_112x112/arcface_celeb_list_new.h5

    