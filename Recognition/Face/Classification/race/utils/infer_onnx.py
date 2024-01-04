import onnxruntime

import argparse

import cv2
import numpy as np
import torch
import os
import pickle
import time

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def read_image(img):
    img = cv2.resize(img, (112, 112))

    img = np.transpose(img, (2, 0, 1)) # 112,112,3 -> 3,112,112
    img = torch.from_numpy(img).unsqueeze(0).float() # 3,112,112 -> 1,3,112,112
    img.div_(255).sub_(0.5).div_(0.5) # ( img/255 - 0.5 )/0.5
    
    img = to_numpy(img)

    return img

# softmax 로 최종 나이,성별 계산
def get_pred(sf_gender, sf_age):#,resize):
    # sf_gender : 성별에 대한 softmax / shape (1,2) / 0:female 1:male
    # sf_age    : 나이에 대한 softmax / shape (1,202) / [0:101]:female [101:202]:male

    age_female = sf_age[:,:101] # female 일때 age softmax 값 
    age_male = sf_age[:,101:]   # male 일때 age softmax 값 

    #p_female = np.reshape(sf_gender[:,0],(resize,1))*age_female
    #p_male = np.reshape(sf_gender[:,1],(resize,1))*age_male
    p_female = sf_gender[:,0][0]*age_female # female 일때 predict age = female gender softmax * female age softmax
    p_male = sf_gender[:,1][0]*age_male     # male 일때 predict age  = male gender softmax * male age softmax
    p_age = p_female+p_male                 # 최종 predict age

    pred_gender = np.argmax(sf_gender)#,axis=1)
    pred_age = np.argmax(p_age)#,axis=1)

    return pred_gender, pred_age


@torch.no_grad()
def test(onnx_path,path):

    # onnx load
    ort_session = onnxruntime.InferenceSession(onnx_path)


    label_total=np.zeros(len(label_list))
    gender_total=np.zeros(2)
    age_total=np.zeros(8)

    label_ct=np.zeros(len(label_list))
    gender_ct=np.zeros(2)
    age_ct=np.zeros(8)

    all_time=[]

    # data load
    d = h5py.File(path,'r')

    images = d["image"][()]
    ages = d['age'][()]
    genders = d["gender"][()]

    total = len(ages)


    mae=0

    ca3=0
    ca5=0

    for li in range(len(ages)):
        if li%1000==0: print(li)

        image = images[li]

        age = int(float(ages[li]))
        gender = int(float(genders[li]))

        img = read_image(image)

        start = time.time()

        # onnx running
        sf_gender, sf_age = ort_session.run(None, {ort_session.get_inputs()[0].name:img})

        pred_g,pred_a = get_pred(sf_gender,sf_age)#,img.shape[0])

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
    parser = argparse.ArgumentParser(description='race onnx model inference')
    parser.add_argument('--onnx_path', type=str, default='')
    parser.add_argument('--h5_path', type=str, default="dlib_aligned_112x112_nomargin/kceleb_dlib_aligned.h5")
    args = parser.parse_args()
    test(args.onnx_path, args.h5_path)
