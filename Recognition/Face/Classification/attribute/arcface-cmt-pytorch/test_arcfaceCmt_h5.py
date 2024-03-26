import argparse

import cv2
import numpy as np
import torch
import os
import pickle
import time

from backbones import get_cmt_model, get_cmt_model_mbf, get_cmt_model_res100,get_group_model
import random
import h5py
import torch.nn as nn

from sklearn.metrics import mean_absolute_error
from collections import OrderedDict

from torchvision import transforms as T


label_list = ['1_1','1_2','1_3','1_4','1_5','1_6','1_7','1_8',
                    '2_1','2_2','2_3','2_4','2_5','2_6','2_7','2_8']
label_age = {'1':[0,2],'2':[3,13],'3':[14,18],'4':[19,28],'5':[29,45],'6':[46,58],'7':[59,77],'8':[78,100]}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def match_lb(age,gender):

    label_age = {'1':[0,2],'2':[3,13],'3':[14,18],'4':[19,28],'5':[29,45],'6':[46,58],'7':[59,77],'8':[78,100]}

    if gender==0:
        lb_format = "2_{}"
    else:
        lb_format = "1_{}"
        
    for k in label_age.keys():
        s=label_age[k][0]
        e=label_age[k][1]
        if s<=age<=e:
            match = lb_format.format(k)
            break
            
    return match


def read_image(img):

    img = cv2.resize(img, (112, 112))

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return img

def get_pred(sf_gender, sf_age):#,resize):
    age_female = sf_age[:,:101]
    age_male = sf_age[:,101:]


    #p_female = np.reshape(sf_gender[:,0],(resize,1))*age_female
    #p_male = np.reshape(sf_gender[:,1],(resize,1))*age_male

    p_female = sf_gender[:,0][0]*age_female
    p_male = sf_gender[:,1][0]*age_male
    p_age = p_female+p_male

    pred_gender = np.argmax(sf_gender)#,axis=1)
    pred_age = np.argmax(p_age)#,axis=1)


    return pred_gender, pred_age


@torch.no_grad()
def test(args):


    weight, name, path, backbone_name, embedding_size  = args.weight, args.network, args.h5_path, args.backbone_name, args.embedding_size

    if backbone_name=='res50':
        net = get_cmt_model("", embedding_size)
    elif backbone_name=='res100':
        net = get_cmt_model_res100("",embedding_size)
    elif backbone_name=='mbf':
        net = get_cmt_model_mbf("",True,embedding_size)
    elif backbone_name=='g50_ag':
        st_type, num_inter, groups, mode = args.st_type, args.num_inter, args.groups, args.mode
        net = get_group_model(name='r50',
                                pretrained_path = False,
                                st_type=st_type,
                                freeze=False,
                                num_embeddings=embedding_size,
                                num_intermediate=num_inter,
                                groups=groups,
                                mode=mode,
                                fp16=True
                                        )
    else:
        print("wrong backbone name, exit")
        exit()

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

    label_total=np.zeros(len(label_list))
    gender_total=np.zeros(2)
    age_total=np.zeros(8)

    label_ct=np.zeros(len(label_list))
    gender_ct=np.zeros(2)
    age_ct=np.zeros(8)

    all_time=[]

    d = h5py.File(path,'r')

    images = d["image"]
    if "appa" in path:
        ages = d['age_appa']
    else:   
        ages = d['age']
    genders = d["gender"]

    total = len(ages)


    mae=0

    ca3=0
    ca5=0

    with torch.no_grad():
        for li in range(len(ages)):
            if li%1000==0: print(li)

            image = images[li]

            age = int(float(ages[li]))
            gender = int(float(genders[li]))

            img = read_image(image)

            img = img.to(device)
            start = time.time()

            if backbone_name=='g50_ag':
                if st_type==1:
                    sf_gender, sf_age = net(img)
                else:
                    sf_g, sf_a, i_g, i_a = net(img)
                    i_g = torch.pow(i_g,1/3)*0.1
                    i_a = torch.pow(i_a,1/3)*0.1

                    sf_gender = sf_g + i_g
                    sf_age = sf_a + i_a

                if args.norm:
                    sf_gender = sf_gender/torch.linalg.norm(sf_gender)
                    sf_age = sf_age/torch.linalg.norm(sf_age)

            else:
                sf_gender, sf_age = net(img)

            sf_gender = sf_gender.cpu().numpy()
            sf_age = sf_age.cpu().numpy()

            pred_g,pred_a = get_pred(sf_gender,sf_age)#,img.size(0))

            mae += mean_absolute_error(np.array([age]),np.array([pred_a]))

            end = time.time()

            infer_time = (end-start)
            all_time.append(infer_time)


            match = match_lb(age,gender)
            match_age = match[-1]
            match_gender = match[0]
            pred_match = match_lb(pred_a,pred_g)
            pred_age = pred_match[-1]
            pred_gender = pred_match[0]

            label_total[label_list.index(match)]+=1
            gender_total[int(match_gender)-1]+=1
            age_total[int(match_age)-1]+=1

            # gender
            if pred_gender==match_gender:
                gender_ct[int(match_gender)-1]+=1
            # age
            if pred_age==match_age:
                age_ct[int(match_age)-1]+=1
            if (pred_gender==match_gender) and (pred_age==match_age):
                label_ct[label_list.index(match)]+=1

            # age ca3
            if abs(age-pred_a)<3:
                ca3+=1
            # age ca5
            if abs(age-pred_a)<5:
                ca5+=1

    label_acc = sum(label_ct)/sum(label_total)
    gender_acc = sum(gender_ct)/sum(gender_total)
    male_acc = gender_ct[0]/gender_total[0]
    female_acc = gender_ct[1]/gender_total[1]
    age_acc = sum(age_ct)/sum(age_total)

    final_mae = mae/total
    ca3_acc = ca3/total
    ca5_acc = ca5/total


    print("Inference time (total time/ total len) (s):",sum(all_time)/total)
    print("label acc:",label_acc)
    print("gender acc:{}, \nfemale_acc:{}, male_acc:{}".format(gender_acc,female_acc,male_acc))
    print("age_acc:",age_acc)
    print("MAE:{0:.4f}".format(final_mae))
    print("CA3:{0:.4f}".format(ca3_acc))
    print("CA5:{0:.4f}".format(ca5_acc))

    for i in range(len(label_list)):
        print("{} acc:{}".format(label_list[i],label_ct[i]/label_total[i]))
    print()
    for i in range(1,9):
        print("{}_{} acc:{}".format(i,label_age[str(i)],age_ct[i-1]/age_total[i-1]))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='ms1mv3_new_agengender_211110_nf_aligned/best/epoch_4_step_8295_val_loss_0.6511_val_acc_0.7838_best.pth')
    parser.add_argument('--h5_path', type=str, default="/data/notebook/NAS/Gender-Age/test_data/dlib_aligned_112x112/test_wiki_dlib_aligned.h5")
    parser.add_argument('--backbone_name',type=str,default='res50',help='res50, res100, mbf')
    parser.add_argument("--embedding_size",type=int,default=512)

    # for group face
    parser.add_argument("--st_type",type=int,default=1,help="group face model type")
    parser.add_argument("--num_inter",type=int,default=256)
    parser.add_argument("--groups",type=int,default=32)
    parser.add_argument("--mode",type=str,default='S')
    parser.add_argument("--norm",type=lambda x: (str(x).lower() == 'true'),default=False)
    #

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

    