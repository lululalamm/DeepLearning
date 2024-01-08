# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import os
import cv2
import torch
from torchvision import datasets
from torchvision.transforms import Compose
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
import psutil

import h5py

import time


def opencv_loader(path):
    img = cv2.imread(path)
    return img


target_dict={'real':0,'print':1,'replay':2,'3d':3}

class DatasetFolderFT_h5(Dataset):
    def __init__(self, h5_path,transform=None, target_transform=None,ft_width=10,ft_height=10,ycrcb=False):

        self.h5_path = h5_path

        self.transform = transform
        self.target_transform = target_transform
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.ycrcb = ycrcb
        print("ycrcb:",self.ycrcb)
        
        d = h5py.File(h5_path,'r')

        self.samples = d['image']
        self.targets = d['target']

        print("total:",len(self.samples))
        print("DatasetFolderFT_h5 init")

    def __len__(self):
        return len(self.samples)-1

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample is None:
            print('image is None --> ', index)
        # sample = np.squeeze(sample,axis=0)
        target = int(self.targets[index])
        if not target==0:
            target=1

        #print(sample.shape)
        ft_sample = generate_FT(sample)
        if ft_sample is None:
            print('FT image is None -->', index)
        assert sample is not None
        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)
        # ft_sample = generates_FT(sample)
        #sample = torch.from_numpy(sample).float()

        if self.ycrcb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2YCR_CB)
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, index)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target

        # sample = self.samples[index]
        # return sample

class DatasetFolderFT_h5_sampler(Dataset):
    def __init__(self, h5_path,transform=None, target_transform=None,ft_width=10,ft_height=10,ycrcb=False):

        self.h5_path = h5_path

        self.transform = transform
        self.target_transform = target_transform
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.ycrcb = ycrcb
        print("ycrcb:",self.ycrcb)

        d = h5py.File(h5_path,'r')

        self.samples = d['image']
        self.targets = d['target']

        print("total:",len(self.samples))
        print("DatasetFolderFT_h5 init")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]

        ft_sample = generates_FT(sample)

        if self.ycrcb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2YCR_CB)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, ft_sample, target

    def _permute_tf_to_torch(self, tensor):
        """Function to load PIL images in correct format required by PyTorch
        This extends the capabiliy of torchvision.transforms.ToTensor to 4D arrays
        """
        return tensor.permute([0, 3, 2, 1])

    def _from_numpy(self, tensor):
        return torch.from_numpy(tensor).float()

class DatasetFolderFT_txt(Dataset):
    def __init__(self, txt_path, new_base,transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader,ycrcb=False):

        self.txt_path = txt_path
        self.new_base = new_base
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.samples=[]
        self.samples_FT=[]
        self.targets=[]

        self.ycrcb = ycrcb
        print("ycrcb:",self.ycrcb)

        count={0:0,1:0,2:0,3:0}
        lines = open(self.txt_path,'r').readlines()
        for line in tqdm(lines):
            sp = line.strip().split(",")
            _,org_path,_,_,_,_,_,mask,_,label = sp
            if mask=='Y':
                continue
            
            new_path = os.path.join(self.new_base,os.path.join(*org_path.split("/")[-2:]))
            target = target_dict[label]
            count[target]+=1

            img = cv2.imread(new_path)
            self.samples.append(img)
            self.targets.append(target)

            ft_img = generate_FT(img)
            ft_img = cv2.resize(ft_img, (self.ft_width, self.ft_height))
            ft_img = torch.from_numpy(ft_img).float()
            ft_img = torch.unsqueeze(ft_img, 0)
            self.samples_FT.append(ft_img)

        print("DatasetFolderFT_txt init")
        print("target count:",count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.samples[index]
        target = self.targets[index]
        ft_sample = self.samples_FT[index]

        if self.ycrcb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2YCR_CB)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target

class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader,ycrcb=False):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.ycrcb = ycrcb
        print("DatasetFolderFT init")
        print("ycrcb:",self.ycrcb)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.ycrcb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2YCR_CB)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, ft_sample, target


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg

def generates_FT(images,ft_width=10,ft_height=10):
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        fimg = np.log(np.abs(fshift)+1)
        maxx = -1
        minn = 100000
        for i in range(len(fimg)):
            if maxx < max(fimg[i]):
                maxx = max(fimg[i])
            if minn > min(fimg[i]):
                minn = min(fimg[i])
        fimg = (fimg - minn+1) / (maxx - minn+1)

        fimg = cv2.resize(fimg, (ft_width, ft_height))
        fimg = torch.from_numpy(fimg).float()
        fimg = torch.unsqueeze(fimg, 0)
    return fimg
