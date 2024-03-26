import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms as T

import random
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
import cv2

import h5py
from sklearn.model_selection import train_test_split
from typing import Callable




# gamma LUT table
def gamma_lut(gamma):
    invGamma = 1.0 / gamma
    return np.array([((i/255.0)**invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')

def gamma_LUT(img, num): 
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    gamma_S = cv2.LUT(S, gamma_lut(num))
    gamma_img =cv2.merge([H, gamma_S, V])
    gamma_img = cv2.cvtColor(gamma_img, cv2.COLOR_HSV2BGR)
    return gamma_img

def gamma_LUT(img,alpha, num): 
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    if num == 1:
        H += 10
        H = cv2.LUT(H, gamma_lut(alpha))
    
    else:
        S = cv2.LUT(S, gamma_lut(alpha))
        
    lut_img =cv2.merge([H, S, V])
    lut_img = cv2.cvtColor(lut_img, cv2.COLOR_HSV2BGR)
    return lut_img
#


def image_augmentation(img, num):
    if num==0: # brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.8)
    elif num==1: # horizontal flip
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif num==2: # horizontal shift
        w,h = img.size
        shift = random.randint(0,int(w*0.2))
        img = ImageChops.offset(img, shift, 0)
        img.paste((0), (0, 0, shift, h))
    elif num==3: # vertical shift
        w,h = img.size
        shift = random.randint(0,int(h*0.2))
        img = ImageChops.offset(img, 0, shift)
        img.paste((0), (0, 0, w, shift))
    elif num==4: # rotation
        img = img.rotate(random.randint(-30,30))
    elif num==5: # shearing
        cx, cy = 0, random.uniform(0.0, 0.3)
        img = img.transform(img.size, method=Image.AFFINE, data=[1, cx, 0, cy, 1, 0,])
    elif num==6: # zoom
        zoom = random.uniform(0.7, 1.3) #0.7 ~ 1.3
        w, h = img.size
        x = w / 2
        y = h / 2
        img = img.crop((x - (w/ 2 / zoom), y - (h / 2 / zoom), x + (w / 2 / zoom), y + (h/ 2 / zoom)))
        img = img.resize((w, h), Image.LANCZOS)
    elif num==7: # change hue or saturation, Gamma Lut
        alpha = (random.random()*1.5)+0.5
        c_num = random.sample([1,2], 1)

        image = np.array(img)
        gamma_img = gamma_LUT(image,alpha, num)
        img = Image.fromarray(gamma_img)
    elif num==8: # gaussianBlur
        gaussianBlur = ImageFilter.GaussianBlur(2)
        img = img.filter(gaussianBlur)
    
    ## equalize
    ##img = ImageOps.equalize(img)
    return img  

def get_lb(age,gender):
    range_age = [[0,2],[3,13],[14,18],[19,28],[29,45],[46,58],[59,77],[78,100]]
    labels = ["1_1","1_2","1_3","1_4","1_5","1_6","1_7","1_8",
                "2_1","2_2","2_3","2_4","2_5","2_6","2_7","2_8"]
    
    for i,alist in enumerate(range_age):
        if age>=alist[0] and age<=alist[1]:
            age_lb = i+1
            break
    gender_lb = '1' if gender==1 else '2'
    
    lb = "{}_{}".format(gender_lb,str(age_lb))
    lb_idx = labels.index(lb)
    
    return lb_idx

def load_data_hdf5(data_path,convert=""):
    d = h5py.File(data_path,'r')

    images = d["image"][()]
    ages = d['age'][()]
    genders = d["gender"][()]
    img_size = d["img_size"][()]

    labels=[]
    for i in range(len(ages)):
        lb = get_lb(ages[i],genders[i])
        labels.append(lb)
        
    labels = np.array(labels)

    labels_uniq, labels_ct = np.unique(labels,return_counts=True)
    one_ct = np.where(labels_ct==1)[0]

    image_add=[]
    label_add=[]

    for o in one_ct:
        one_label = labels_uniq[o]
        label_idx = np.where(labels==one_label)[0][0]

        image_add.append(images[label_idx])
        label_add.append(labels[label_idx])

        images = np.delete(images,label_idx,axis=0)
        labels = np.delete(labels,label_idx)

    x_train,x_val,y_train,y_val = train_test_split(images,labels,train_size=0.8,stratify=labels)

    if image_add:
        x_train = np.concatenate((x_train,image_add),axis=0)
        y_train = np.concatenate((y_train,label_add),axis=0)

    if convert=='rgb' or convert=='bgr':
        x_train = x_train[...,::-1]
        y_train = y_train[...,::-1]


    return x_train,x_val,y_train,y_val

def load_data_hdf5_tv(data_path):
    d = h5py.File(data_path,'r')

    images = d["image"][()]
    ages = d['age'][()]
    genders = d["gender"][()]
    img_size = d["img_size"][()]

    labels=[]
    for i in range(len(ages)):
        lb = get_lb(ages[i],genders[i])
        labels.append(lb)
        
    labels = np.array(labels)

    return images,labels

def load_data_hdf5_202(data_path,convert=""):
    d = h5py.File(data_path,'r')

    images = d["image"][()]
    ages = d['age'][()]
    genders = d["gender"][()]
    img_size = d["img_size"][()]

    labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = ages[i]  if genders[i]==0 else ages[i]+101
        labels.append(lb)
        
    labels = np.array(labels)

    labels_uniq, labels_ct = np.unique(labels,return_counts=True)
    one_ct = np.where(labels_ct==1)[0]

    image_add=[]
    label_add=[]

    for o in one_ct:
        one_label = labels_uniq[o]
        label_idx = np.where(labels==one_label)[0][0]

        image_add.append(images[label_idx])
        label_add.append(labels[label_idx])

        images = np.delete(images,label_idx,axis=0)
        labels = np.delete(labels,label_idx)

    x_train,x_val,y_train,y_val = train_test_split(images,labels,train_size=0.8,stratify=labels)

    if image_add:
        x_train = np.concatenate((x_train,image_add),axis=0)
        y_train = np.concatenate((y_train,label_add),axis=0)

    if convert=='rgb' or convert=='bgr':
        x_train = x_train[...,::-1]
        y_train = y_train[...,::-1]

    return x_train,x_val,y_train,y_val

def load_data_hdf5_202_tv(data_path,convert=""):
    d = h5py.File(data_path,'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]

    labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = ages[i]  if genders[i]==0 else ages[i]+101
        labels.append(lb)
        
    labels = np.array(labels)


    return images,labels

def load_data_hdf5_202_tv_concat(data_paths,convert=""):

    d = h5py.File(data_paths[0],'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]

    for data in data_paths[1:]:

        d = h5py.File(data,'r')

        images = np.concatenate((images,d["image"])) 
        ages = np.concatenate((ages,d['age']))
        genders = np.concatenate((genders,d["gender"]))

    if convert=='rgb' or convert=='bgr':
        images = images[...,::-1]

    labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = ages[i]  if genders[i]==0 else ages[i]+101
        labels.append(lb)
        
    labels = np.array(labels)


    return images,labels

def get_idx_20(age,gender):
    if age>99:
        idx = 9
    else:
        idx = age//10
    if gender==1:
        idx += 10
    return idx

def get_idx_14(age,gender):
    idx = age//10
    if idx<=1:
        idx=0
    elif idx>=7:
        idx=6
    else:
        idx -= 1
    if gender==1:
        idx += 7
    return idx

def get_idx_18(age,gender):
    return

def load_data_hdf5_18_tv(data_path,convert=""):
    d = h5py.File(data_path,'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]

    labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = get_idx_18(ages[i],genders[i])
        labels.append(lb)
        
    labels = np.array(labels)


    return images,labels

def load_data_hdf5_14_tv(data_path,convert=""):
    d = h5py.File(data_path,'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]

    labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = get_idx_14(ages[i],genders[i])
        labels.append(lb)
        
    labels = np.array(labels)


    return images,labels

def load_data_hdf5_14_tv_concat(data_paths,convert=""):
    d = h5py.File(data_paths[0],'r')

    images  = []
    ages = []
    genders = []

    for data in data_paths:

        d = h5py.File(data,'r')

        images_ = d['image']
        ages_ = d['age']
        genders_ = d['gender']

        for i in range(ages_.shape[0]):
            images.append(images_[i])
            ages.append(ages_[i])
            genders.append(genders_[i])
        
        print("load and append:",data)

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]

    images = np.array(images,dtype=np.uint8)
    ages = np.array(ages,dtype=np.uint8)
    genders = np.array(genders,dtype=np.uint8)
        
    labels=[]
    for i in range(len(ages)):
        lb = get_idx_14(ages[i],genders[i])
        labels.append(lb)
    labels = np.array(labels)
    
    return images,labels


def load_data_hdf5_20_tv(data_path,convert=""):
    d = h5py.File(data_path,'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]

    labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = get_idx_20(ages[i],genders[i])
        labels.append(lb)
        
    labels = np.array(labels)


    return images,labels

def load_data_hdf5_20_tv_concat(data_paths,convert=""):
    d = h5py.File(data_paths[0],'r')

    images  = []
    ages = []
    genders = []

    for data in data_paths:

        d = h5py.File(data,'r')

        images_ = d['image']
        ages_ = d['age']
        genders_ = d['gender']

        for i in range(ages_.shape[0]):
            images.append(images_[i])
            ages.append(ages_[i])
            genders.append(genders_[i])
        
        print("load and append:",data)

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]

    images = np.array(images,dtype=np.uint8)
    ages = np.array(ages,dtype=np.uint8)
    genders = np.array(genders,dtype=np.uint8)
        
    labels=[]
    for i in range(len(ages)):
        lb = get_idx_20(ages[i],genders[i])
        labels.append(lb)
    labels = np.array(labels)
    
    return images,labels

def load_data_hdf5_202_kfold(data_path):

    out_values=[]

    d = h5py.File(data_path,'r')

    images = d["image"][()]
    ages = d['age'][()]
    genders = d["gender"][()]
    img_size = d["img_size"][()]

    #labels=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages)):
        lb = ages[i]  if genders[i]==0 else ages[i]+101
        #labels.append(lb)
        out_values.append([images[i],lb])
        
    #labels = np.array(labels)

    return np.array(out_values)

def load_data_hdf5_202_tv2(data_path):
    d = h5py.File(data_path,'r')

    # age -> uid
    # gender -> age
    # uid -> gender

    images_train = d["img_train"][()]
    images_val = d["img_valid"][()]

    ages_train = d["age_train"][()]
    ages_val = d["age_valid"][()]
    genders_train = d['gender_train'][()]
    genders_val = d['gender_valid'][()]


    labels_train=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages_train)):
        lb = ages_train[i]  if genders_train[i]==0 else ages_train[i]+101
        labels_train.append(lb)
    labels_train = np.array(labels_train)

    labels_val=[]
    # age+101, (0)female+0 , (1)male+101
    for i in range(len(ages_val)):
        lb = ages_val[i]  if genders_val[i]==0 else ages_val[i]+101
        labels_val.append(lb)
    labels_val = np.array(labels_val)


    return images_train,images_val,labels_train,labels_val


# c3ae

'''
def get_age_dist(age,interval=10,category=12,elips=0.000001):
    embed = [0 for x in range(0, category)]
    
    right_prob = age % interval * 1.0 / interval
    left_prob = 1 - right_prob

    idx = age // interval


    if left_prob:
        embed[idx] = left_prob
    if right_prob and idx + 1 < category:
        embed[idx+1] = right_prob
        
    return embed

def load_data_hdf5_c3ae(data_path,convert=""):
    
    d = h5py.File(data_path,'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]
    
    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]
        
    lb_age_dist=[]
    lb_age_const=[]
    lb_gender=[]

    for i in range(len(ages)):
        lb_age_dist.append(get_age_dist(ages[i]))
        lb_age_const.append(ages[i])
        lb_gender.append(genders[i])
        
    lb_age_dist = np.array(lb_age_dist)
    lb_age_const = np.array(lb_age_const,dtype=np.uint8)
    lb_gender = np.array(lb_gender,dtype=np.uint8)
    
    return images,lb_age_dist, lb_age_const, lb_gender

def load_data_hdf5_c3ae_concat(data_paths,convert=""):
    
        
    lb_age_dist=[]
    lb_age_const=[]
    lb_gender=[]
    images  = []

    for data in data_paths:

        d = h5py.File(data_paths[0],'r')

        images_ = d["image"]
        ages = d['age']
        genders = d["gender"]
        img_size = d["img_size"]


        for i in range(len(ages)):
            images.append(images_[i])
            lb_age_dist.append(get_age_dist(ages[i]))
            lb_age_const.append(ages[i])
            lb_gender.append(genders[i])

        print("load and append:",data)

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]
    
    images = np.array(images,dtype=np.uint8)
    lb_age_dist = np.array(lb_age_dist)
    lb_age_const = np.array(lb_age_const,dtype=np.uint8)
    lb_gender = np.array(lb_gender,dtype=np.uint8)
    
    return images,lb_age_dist, lb_age_const, lb_gender
'''
def get_age_dist_c20(age,category=11):
    embed = [0 for x in range(0, category)]
    interval_list =[[0,10],[10,10],[20,10],[30,10],[40,10],[50,10],[60,10],[70,10],[80,10],[90,20]]
    # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110
    
    if age>110: age=110
    if age<0: age==0
    
    interval_idx = age//10
    if interval_idx>9:
        interval_idx=9
        
    interval = interval_list[interval_idx]
    
    right_prob = (age-interval[0])/interval[1]
    left_prob = (interval[1]-(age-interval[0]))/interval[1]
    
    if left_prob:
        embed[interval_idx]=left_prob
    if right_prob and interval_idx+1 < category:
        embed[interval_idx+1]=right_prob

    return embed

def get_age_dist_c14(age,category=8):
    embed = [0 for x in range(0, category)]
    interval_list =[[0,20],[20,10],[30,10],[40,10],[50,10],[60,10],[70,40]]
    # 0, 20, 30, 40, 50, 60, 70, 110
    
    if age>110: age=110
    if age<0: age==0
    
    interval_idx = age//10
    if interval_idx<=1:
        interval_idx=0
    elif interval_idx>=7:
        interval_idx=6
    else:
        interval_idx-=1
        
    interval = interval_list[interval_idx]
    
    right_prob = (age-interval[0])/interval[1]
    left_prob = (interval[1]-(age-interval[0]))/interval[1]
    
    if left_prob:
        embed[interval_idx]=left_prob
    if right_prob and interval_idx+1 < category:
        embed[interval_idx+1]=right_prob

    return embed

def load_data_hdf5_c3ae(data_path,class_num,convert=""):
    
    d = h5py.File(data_path,'r')

    images = d["image"]
    ages = d['age']
    genders = d["gender"]
    img_size = d["img_size"]
    
    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]
        
    lb_age_dist=[]
    lb_age_const=[]
    lb_gender=[]

    for i in range(len(ages)):
        if class_num==14:
            lb_age_dist.append(get_age_dist_c14(ages[i]))
        elif class_num==20:
            lb_age_dist.append(get_age_dist_c20(ages[i]))
        else:
            print("class num error")
            return
        lb_age_const.append(ages[i])
        lb_gender.append(genders[i])
        
    lb_age_dist = np.array(lb_age_dist)
    lb_age_const = np.array(lb_age_const,dtype=np.uint8)
    lb_gender = np.array(lb_gender,dtype=np.uint8)
    
    return images,lb_age_dist, lb_age_const, lb_gender

def load_data_hdf5_c3ae_concat(data_paths,class_num,convert=""):
    
        
    lb_age_dist=[]
    lb_age_const=[]
    lb_gender=[]
    images  = []

    for data in data_paths:

        d = h5py.File(data_paths[0],'r')

        images_ = d["image"]
        ages = d['age']
        genders = d["gender"]
        img_size = d["img_size"]


        for i in range(len(ages)):
            images.append(images_[i])
            if class_num==14:
                lb_age_dist.append(get_age_dist_c14(ages[i]))
            elif class_num==20:
                lb_age_dist.append(get_age_dist_c20(ages[i]))
            else:
                print("class num error")
                return
            lb_age_const.append(ages[i])
            lb_gender.append(genders[i])

        print("load and append:",data)

    if convert=='rgb' or convert=='bgr':
        images = np.array(images)[...,::-1]
    
    images = np.array(images,dtype=np.uint8)
    lb_age_dist = np.array(lb_age_dist)
    lb_age_const = np.array(lb_age_const,dtype=np.uint8)
    lb_gender = np.array(lb_gender,dtype=np.uint8)
    
    return images,lb_age_dist, lb_age_const, lb_gender

class AgeNGender_Dataset_aug_c3ae(Dataset):
    def __init__(self, set_x,set_y_dist,set_y_const,set_y_gender,image_size,mode):


        self.transform = T.Compose([
                T.ToPILImage(),
                ])

        self.transform2 = T.GaussianBlur(kernel_size=3)

        self.transform3 = T.RandomRotation(degrees=(-15,15),interpolation=2)

        self.transform_final = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    


        self.transforms_val = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        self.image_list = set_x
        self.lblist_dist = set_y_dist
        self.lblist_const = set_y_const
        self.lblist_gender = set_y_gender

        self.num_images = len(set_x)
        self.image_size = image_size

        self.mode = mode
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        lb_dist = self.lblist_dist[idx]
        lb_const = self.lblist_const[idx]
        lb_gender = self.lblist_gender[idx]

        # random augmentation
        if self.mode=='train_pre':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            image = self.transform_final(image)
        elif self.mode=='train_add':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            if random.random()>0.5:
                image = self.transform2(image)
            image = self.transform_final(image)

        elif self.mode=='val':
            image = self.transforms_val(image)

        return image.float(), lb_dist, lb_const, lb_gender

class AgeNGender_Dataset_c3ae(Dataset):
    def __init__(self, set_x,set_y_dist,set_y_const,set_y_gender,image_size,mode):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        self.image_list = set_x
        self.lblist_dist = set_y_dist
        self.lblist_const = set_y_const
        self.lblist_gender = set_y_gender

        self.num_images = len(set_x)
        self.image_size = image_size

        self.mode = mode
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        lb_dist = self.lblist_dist[idx]
        lb_const = self.lblist_const[idx]
        lb_gender = self.lblist_gender[idx]

        if self.transform:
            image = self.transform(image)

        return image.float(), lb_dist, lb_const, lb_gender

class AgeNGender_Dataset_aug_c3ae_idx(Dataset):
    def __init__(self, set_x,set_y_dist,set_y_const,set_y_gender,set_idx,image_size,mode):


        self.transform = T.Compose([
                T.ToPILImage(),
                ])

        self.transform2 = T.GaussianBlur(kernel_size=3)

        self.transform3 = T.RandomRotation(degrees=(-15,15),interpolation=2)

        self.transform_final = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    


        self.transforms_val = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        self.image_list = set_x
        self.lblist_dist = set_y_dist
        self.lblist_const = set_y_const
        self.lblist_gender = set_y_gender
        self.idx_list = set_idx

        self.num_images = len(set_x)
        self.image_size = image_size

        self.mode = mode
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        lb_dist = self.lblist_dist[idx]
        lb_const = self.lblist_const[idx]
        lb_gender = self.lblist_gender[idx]
        idx_val = self.idx_list[idx]

        # random augmentation
        if self.mode=='train_pre':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            image = self.transform_final(image)
        elif self.mode=='train_add':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            if random.random()>0.5:
                image = self.transform2(image)
            image = self.transform_final(image)

        elif self.mode=='val':
            image = self.transforms_val(image)

        return image.float(), lb_dist, lb_const, lb_gender, idx_val

class AgeNGender_Dataset_c3ae_idx(Dataset):
    def __init__(self, set_x,set_y_dist,set_y_const,set_y_gender,set_idx,image_size,mode):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        self.image_list = set_x
        self.lblist_dist = set_y_dist
        self.lblist_const = set_y_const
        self.lblist_gender = set_y_gender
        self.idx_list = set_idx

        self.num_images = len(set_x)
        self.image_size = image_size

        self.mode = mode
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        lb_dist = self.lblist_dist[idx]
        lb_const = self.lblist_const[idx]
        lb_gender = self.lblist_gender[idx]
        idx_val = self.idx_list[idx]

        if self.transform:
            image = self.transform(image)

        return image.float(), lb_dist, lb_const, lb_gender, idx_val




class AgeNGender_Dataset(Dataset):
    def __init__(self, set_x,set_y,image_size):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        
        self.image_list = set_x
        self.lb_list = set_y

        self.num_images = len(set_y)
        self.image_size = image_size
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        #image = Image.open(self.image_list[idx])
        #image = image.resize((self.image_size, self.image_size))

        # random augmentation 
        #if random.uniform(0,1)>0.5:
        #    image = image_augmentation(image, random.randint(0,7)) ## blur 넣으면 (0,8)

        #image = image.convert('RGB')

        image = self.image_list[idx]
        label = self.lb_list[idx]


        if self.transform:
            image = self.transform(image)

            
        return image.float(), label


class AgeNGender_Dataset_cvt(Dataset):
    def __init__(self, set_x,set_y,image_size):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        
        self.image_list = set_x
        self.lb_list = set_y

        self.num_images = len(set_y)
        self.image_size = image_size
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        #image = Image.open(self.image_list[idx])
        #image = image.resize((self.image_size, self.image_size))

        # random augmentation 
        #if random.uniform(0,1)>0.5:
        #    image = image_augmentation(image, random.randint(0,7)) ## blur 넣으면 (0,8)

        #image = image.convert('RGB')

        image = self.image_list[idx]
        label = self.lb_list[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform:
            image = self.transform(image)

            
        return image.float(), label
        


class AgeNGender_Dataset_top(Dataset):
    def __init__(self, set_x,set_y,image_size,convert):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        
        self.image_list = set_x
        self.lb_list = set_y

        self.num_images = len(set_y)
        self.image_size = image_size
        self.convert = convert
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        label = self.lb_list[idx]

        if self.convert:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image = image[:round(image.shape[0]*0.7)][:]
        image = cv2.resize(image,(self.image_size,self.image_size))


        if self.transform:
            image = self.transform(image)

            
        return image.float(), label


class AgeNGender_Dataset_aug(Dataset):
    def __init__(self, set_x,set_y,image_size,mode):

        '''
        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2,0.2,0.4,0.05),
                ])
        
        self.transform2 = T.Compose([
                        T.RandomResizedCrop((100,100),scale=(0.9,0.9)),
                        T.Pad(6),
        ])
        # torchvision 0.9.0
        self.transform3 = T.RandomRotation(degrees=(0,5),interpolation=2)
        '''
        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2,0.2,0.4,0.05)
                ])

        self.transform2 = T.GaussianBlur(kernel_size=3)

        self.transform3 = T.RandomRotation(degrees=(-15,15),interpolation=2)

        self.transform_final = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    


        self.transforms_val = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        
        self.image_list = set_x
        self.lb_list = set_y

        self.num_images = len(set_y)
        self.image_size = image_size

        self.mode = mode
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        label = self.lb_list[idx]

        # random augmentation
        if self.mode=='train_pre':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            image = self.transform_final(image)
        elif self.mode=='train_add':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            if random.random()>0.5:
                image = self.transform2(image)
            image = self.transform_final(image)

        elif self.mode=='val':
            image = self.transforms_val(image)


        '''
        if self.mode=='train':
            image = self.transform(image)
            
            if random.random()>0.8:
                image = self.transform2(image)
            
            if random.random()>0.5:
                image = self.transform3(image)

            image = self.transform_final(image)
        
        else:
            image = self.transforms_val(image)
        '''
        return image.float(), label



class AgeNGender_Dataset_kfold(Dataset):
    def __init__(self, dataset,image_size):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
    
        
        self.image_list = dataset[:,0]
        self.lb_list = dataset[:,1]

        self.num_images = len(self.lb_list)
        self.image_size = image_size
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        label = self.lb_list[idx]


        if self.transform:
            image = self.transform(image)

            
        return image.float(), label

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        train_set,
        weights,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(train_set))) if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset

        self.weights = torch.DoubleTensor(weights)


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


# way 1 ( add add .. )
def oversampling_norm_c14(x_train,y_train,genders=['female','male']):
    
    lb_uniq = np.unique(y_train)
    lb_counts=np.zeros(len(lb_uniq),dtype=np.int64)
    
    for ui,uval in enumerate(lb_uniq):
        lb_counts[ui]+=len(np.where(y_train==uval)[0])
        
    new_x_train=[]
    new_y_train=[]

    for gen in genders:
        if gen=='female':
            base_ct = lb_counts[:7]
        else: # male
            base_ct = lb_counts[7:]

        scale = np.mean(base_ct)/10

        max_num = int(np.mean(base_ct)*1.3)
        get_norm=sorted(np.random.normal(max_num,scale,7),reverse=True)

        sort_idx = np.argsort(base_ct)[::-1]

        # add add add
        for i,idx in enumerate(sort_idx):
            if not gen=='female':
                idx+=7
            x_idx = x_train[np.where(y_train==idx)[0]]
            norm_val = int(get_norm[i])

            if len(x_idx)>norm_val:
                new_x_idx = random.sample(list(x_idx),norm_val)
            elif len(x_idx)<norm_val:
                iter_num = norm_val//len(x_idx)
                samp_num = norm_val%len(x_idx)

                new_x_idx = []
                for i in range(iter_num):
                    new_x_idx.extend(x_idx)
                if samp_num>0:
                    add_sample = random.sample(list(x_idx),samp_num)
                    new_x_idx.extend(add_sample)
            else:
                new_x_idx = x_idx

            new_x_train.extend(new_x_idx)
            new_y_train.extend(np.ones(len(new_x_idx),dtype=np.int64)*idx)


    return new_x_train, new_y_train


# way 1 ( add add .. )
# way 1 ( add add .. )
def oversampling_norm_c14_idx(x_train,y_train,genders=['female','male']):
    
    lb_uniq = np.unique(y_train)
    lb_counts=np.zeros(len(lb_uniq),dtype=np.int64)
    
    for ui,uval in enumerate(lb_uniq):
        lb_counts[ui]+=len(np.where(y_train==uval)[0])
        
    new_x_train=[]
    new_y_train=[]
    new_id_train=[]

    idx_train = np.array(list(range(len(y_train))))

    for gen in genders:
        if gen=='female':
            base_ct = lb_counts[:7]
        else: # male
            base_ct = lb_counts[7:]

        scale = np.mean(base_ct)/10

        max_num = int(np.mean(base_ct)*1.3)
        get_norm=sorted(np.random.normal(max_num,scale,7),reverse=True)

        sort_idx = np.argsort(base_ct)[::-1]

        # add add add
        for i,idx in enumerate(sort_idx):
            if not gen=='female':
                idx+=7
            x_idx = x_train[np.where(y_train==idx)[0]]
            id_idx = idx_train[np.where(y_train==idx)[0]]
            norm_val = int(get_norm[i])

            if len(x_idx)>norm_val:
                add_sample = np.array(random.sample(list(enumerate(x_idx)),norm_val))
                new_id_idx = id_idx[list(add_sample[:,0])]
                new_x_idx = add_sample[:,1]
                
            elif len(x_idx)<norm_val:
                iter_num = norm_val//len(x_idx)
                samp_num = norm_val%len(x_idx)

                new_x_idx = []
                new_id_idx = []
                for i in range(iter_num):
                    new_x_idx.extend(x_idx)
                    new_id_idx.extend(id_idx)
                
                if samp_num>0:
                    add_sample = np.array(random.sample(list(enumerate(x_idx)),samp_num))
                    add_idx = id_idx[list(add_sample[:,0])]
                    add_x = add_sample[:,1]
                
                    new_x_idx.extend(add_x)
                    new_id_idx.extend(add_idx)
            else:
                new_x_idx = x_idx
                new_id_idx = id_idx

            new_x_train.extend(new_x_idx)
            new_y_train.extend(np.ones(len(new_x_idx),dtype=np.int64)*idx)
            new_id_train.extend(new_id_idx)


    return new_x_train, new_y_train, new_id_train



# way 1 ( add add .. )
def oversampling_norm_c3ae_c14(x_train,y_train_dist,y_train_const,y_train_gender,genders=['female','male']):
    
    new_x_train=[]
    new_y_dist=[]
    new_y_const=[]
    new_y_gender=[]

    for gen in genders:
        if gen=='female':
            idx_list = np.where(y_train_gender==0)[0]
        else:
            idx_list = np.where(y_train_gender==1)[0]
        
        images_list = x_train[idx_list]
        
        age_const_list = y_train_const[idx_list]
        uniq_list = np.unique(age_const_list)
        dist_list = y_train_dist[idx_list]

        lb_counts = np.zeros(len(uniq_list),dtype=np.int64)
        for ui,uval in enumerate(uniq_list):
            lb_counts[ui]+=len(np.where(age_const_list==uval)[0])

        scale = np.mean(lb_counts)/len(lb_counts)
        
        max_num = int(np.mean(lb_counts)*2)
        get_norm=sorted(np.random.normal(max_num,scale,len(lb_counts)),reverse=True)
        
        sort_idx = np.argsort(lb_counts)[::-1]

        # add add add
        for i,idx in enumerate(sort_idx):
            search_idx = np.where(age_const_list==uniq_list[idx])[0]
            x_idx = images_list[search_idx]
            dist_idx = dist_list[search_idx]
            norm_val = int(get_norm[i])

            if len(x_idx)>norm_val:
                add_sample = np.array(random.sample(list(enumerate(x_idx)),norm_val))
                new_x_idx = add_sample[:,1]
                new_dist_idx = dist_idx[list(add_sample[:,0])]

            elif len(x_idx)<norm_val:
                iter_num = norm_val//len(x_idx)
                samp_num = norm_val%len(x_idx)

                new_x_idx = []
                new_dist_idx = []
                for i in range(iter_num):
                    new_x_idx.extend(x_idx)
                    new_dist_idx.extend(dist_idx)


                if samp_num>0:
                    add_sample = np.array(random.sample(list(enumerate(x_idx)),samp_num))
                    add_dist_idx = dist_idx[list(add_sample[:,0])]
                    add_x = add_sample[:,1]

                    new_x_idx.extend(add_x)
                    new_dist_idx.extend(add_dist_idx)

            else:
                new_x_idx = x_idx
                new_dist_idx = dist_idx

            new_x_train.extend(new_x_idx)
            new_y_dist.extend(new_dist_idx)
            new_y_const.extend(np.ones(len(new_x_idx),dtype=np.int64)*uniq_list[idx])
            if gen=='female':
                new_y_gender.extend(np.zeros(len(new_x_idx),dtype=np.int64))
            else:
                new_y_gender.extend(np.ones(len(new_x_idx),dtype=np.int64))

        return new_x_train,new_y_dist,new_y_const,new_y_gender


# way 1 ( add add .. )
# way 1 ( add add .. )
def oversampling_norm_c3ae_c14_idx(x_train,y_train_dist,y_train_const,y_train_gender,genders=['female','male']):
    
    new_x_train=[]
    new_y_dist=[]
    new_y_const=[]
    new_y_gender=[]
    new_id_train=[]

    for gen in genders:
        if gen=='female':
            idx_list = np.where(y_train_gender==0)[0]
        else:
            idx_list = np.where(y_train_gender==1)[0]
        
        images_list = x_train[idx_list]
        
        age_const_list = y_train_const[idx_list]
        uniq_list = np.unique(age_const_list)
        dist_list = y_train_dist[idx_list]

        lb_counts = np.zeros(len(uniq_list),dtype=np.int64)
        for ui,uval in enumerate(uniq_list):
            lb_counts[ui]+=len(np.where(age_const_list==uval)[0])

        idx_range = np.array(list(range(len(age_const_list))))
        scale = np.mean(lb_counts)/len(lb_counts)
        
        max_num = int(np.mean(lb_counts)*2)
        get_norm=sorted(np.random.normal(max_num,scale,len(lb_counts)),reverse=True)
        
        sort_idx = np.argsort(lb_counts)[::-1]

        # add add add
        for i,idx in enumerate(sort_idx):
            search_idx = np.where(age_const_list==uniq_list[idx])[0]
            x_idx = images_list[search_idx]
            id_idx = idx_range[search_idx]
            dist_idx = dist_list[search_idx]
            norm_val = int(get_norm[i])

            if len(x_idx)>norm_val:
                add_sample = np.array(random.sample(list(enumerate(x_idx)),norm_val))
                new_id_idx = id_idx[list(add_sample[:,0])]
                new_x_idx = add_sample[:,1]
                new_dist_idx = dist_idx[list(add_sample[:,0])]

            elif len(x_idx)<norm_val:
                iter_num = norm_val//len(x_idx)
                samp_num = norm_val%len(x_idx)

                new_x_idx = []
                new_id_idx = []
                new_dist_idx = []
                for i in range(iter_num):
                    new_x_idx.extend(x_idx)
                    new_id_idx.extend(id_idx)
                    new_dist_idx.extend(dist_idx)

                if samp_num>0:
                    add_sample = np.array(random.sample(list(enumerate(x_idx)),samp_num))
                    add_idx = id_idx[list(add_sample[:,0])]
                    add_dist_idx = dist_idx[list(add_sample[:,0])]
                    add_x = add_sample[:,1]

                    new_x_idx.extend(add_x)
                    new_id_idx.extend(add_idx)
                    new_dist_idx.extend(add_dist_idx)

            else:
                new_x_idx = x_idx
                new_id_idx = id_idx
                new_dist_idx = dist_idx

            new_x_train.extend(new_x_idx)
            new_y_dist.extend(new_dist_idx)
            new_y_const.extend(np.ones(len(new_x_idx),dtype=np.int64)*uniq_list[idx])
            if gen=='female':
                new_y_gender.extend(np.zeros(len(new_x_idx),dtype=np.int64))
            else:
                new_y_gender.extend(np.ones(len(new_x_idx),dtype=np.int64))
            new_id_train.extend(new_id_idx)

        return new_x_train,new_y_dist,new_y_const,new_y_gender,new_id_train

class AgeNGender_Dataset_aug_idx(Dataset):
    def __init__(self, set_x,set_y,set_idx,image_size,mode):

        self.transform = T.Compose([
                T.ToPILImage(),
                ])

        self.transform2 = T.GaussianBlur(kernel_size=3)

        self.transform3 = T.RandomRotation(degrees=(-15,15),interpolation=2)

        self.transform_final = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    


        self.transforms_val = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        
        self.image_list = set_x
        self.lb_list = set_y
        self.idx_list = set_idx

        self.num_images = len(set_y)
        self.image_size = image_size

        self.mode = mode
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        image = self.image_list[idx]
        label = self.lb_list[idx]
        idx_val = self.idx_list[idx]

        # random augmentation
        if self.mode=='train_pre':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            image = self.transform_final(image)
        elif self.mode=='train_add':
            image = self.transform(image)
            if random.random()>0.5:
                image = self.transform3(image)
            if random.random()>0.5:
                image = self.transform2(image)
            image = self.transform_final(image)

        elif self.mode=='val':
            image = self.transforms_val(image)

        
        return image.float(), label, idx_val


class AgeNGender_Dataset_idx(Dataset):
    def __init__(self, set_x,set_y,set_idx,image_size):


        self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        
        
        self.image_list = set_x
        self.lb_list = set_y
        self.idx_list = set_idx

        self.num_images = len(set_y)
        self.image_size = image_size
            
    def  __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        

        #image = Image.open(self.image_list[idx])
        #image = image.resize((self.image_size, self.image_size))

        # random augmentation 
        #if random.uniform(0,1)>0.5:
        #    image = image_augmentation(image, random.randint(0,7)) ## blur 넣으면 (0,8)

        #image = image.convert('RGB')

        image = self.image_list[idx]
        label = self.lb_list[idx]
        idx_val = self.idx_list[idx]


        if self.transform:
            image = self.transform(image)

            
        return image.float(), label, idx_val


def not_overlap_check(idxs):
    idxs = np.array(idxs)
    for i,idx in enumerate(idxs):
        check_idx = [ x for xi,x in enumerate(idxs) if not xi==i ]
        if idx in check_idx:
            return False
    return True