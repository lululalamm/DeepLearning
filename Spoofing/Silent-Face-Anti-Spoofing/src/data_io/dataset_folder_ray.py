import ray

import os
import cv2
import torch
from torchvision import datasets
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

def line_process(line,new_base,ft_width,ft_height):
    sp = line.strip().split(",")
    _,org_path,_,_,_,_,_,mask,_,label = sp
    
    new_path = os.path.join(new_base,os.path.join(*org_path.split("/")[-2:]))
    target = target_dict[label]
    img = cv2.imread(new_path)

    ft_img = generate_FT(img)
    ft_img = cv2.resize(ft_img, (ft_width, ft_height))
    ft_img = torch.from_numpy(ft_img).float()
    ft_img = torch.unsqueeze(ft_img, 0)

    return [img,target,ft_img]
    

@ray.remote
def data_load(lines,new_base,ft_width,ft_height):
    results=[]
    for line in lines:
        results_id = line_process(line,new_base,ft_width,ft_height)
        results.append(results_id)

    return results

@ray.remote
def line_process2(line,new_base,ft_width,ft_height):
    sp = line.strip().split(",")
    _,org_path,_,_,_,_,_,mask,_,label = sp
    if mask=='Y':
        return []
    else:
        new_path = os.path.join(new_base,os.path.join(*org_path.split("/")[-2:]))
        target = target_dict[label]
        img = cv2.imread(new_path)

        # ft_img = generate_FT(img)
        # ft_img = cv2.resize(ft_img, (ft_width, ft_height))
        # ft_img = torch.from_numpy(ft_img).float()
        # ft_img = torch.unsqueeze(ft_img, 0)

        return [img,target]#,ft_img]


# class DatasetFolderFT_txt_RAY(Dataset):
#     def __init__(self, txt_path, new_base,transform=None, target_transform=None,
#                  ft_width=10, ft_height=10, loader=opencv_loader,ray_batch=10,num_cpus=23):

#         self.txt_path = txt_path
#         self.new_base = new_base
#         self.ft_width = ft_width
#         self.ft_height = ft_height
#         self.loader = loader
#         self.transform = transform
#         self.target_transform = target_transform

#         self.samples=[]
#         self.samples_FT=[]
#         self.targets=[]

#         lines = open(self.txt_path,'r').readlines()
#         result_list=[]
#         new_base_id = ray.put(new_base)
#         ft_width_id = ray.put(ft_width)
#         ft_height_id = ray.put(ft_height)
#         print("Ray load...")
#         count={0:0,1:0,2:0,3:0}
#         for line in tqdm(lines):
#             result = [line_process2.remote(line,new_base_id,ft_width_id,ft_height_id) for ni in range(num_cpus)]
#             for res in result:
#                 if not res: continue
#                 res = ray.get(res)
   
#                 self.samples.append(res[0])
#                 self.targets.append(res[1])
#                 self.samples_FT.append(res[2])

#                 count[int(res[1])]+=1

#         print("total:",len(self.samples))
#         print("count:",count)
#         print("DatasetFolderFT_txt init")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         sample = self.samples[index]
#         target = self.targets[index]
#         ft_sample = self.samples_FT[index]

#         if self.transform is not None:
#             try:
#                 sample = self.transform(sample)
#             except Exception as err:
#                 print('Error Occured: %s' % err, path)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, ft_sample, target

class DatasetFolderFT_txt_RAY(Dataset):
    
    def __init__(self, txt_path, new_base,transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader,ray_batch=10,num_cpus=23,ycrcb=False):

        self.txt_path = txt_path
        self.new_base = new_base
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.ycrcb= ycrcb

        # self.samples=[]
        # self.samples_FT=[]
        # self.targets=[]

        lines = open(self.txt_path,'r').readlines()
        result_list=[]
        new_base_id = ray.put(new_base)
        ft_width_id = ray.put(ft_width)
        ft_height_id = ray.put(ft_height)
        print("Ray load...")
        for i in tqdm(range(0,len(lines),num_cpus*ray_batch)):
            for ni in range(num_cpus):
                start = (ni*ray_batch)+i
                end = ni*ray_batch+(ray_batch-1)+i+1
                
                if start>(len(lines)-1):
                    break
                if end>len(lines):
                    end = len(lines)
                batch_line = lines[start:end]
                result = data_load.remote(batch_line,new_base_id,ft_width_id,ft_height_id)
                result_list.append(result)
        
        print("get")
        s1=time.time()
        self.samples = ray.get(result_list)
        e1=time.time()
        print("time:",(e1-s1),"s")
        self.samples = np.array(self.samples)
        print("reshape")
        s3 = time.time()
        self.samples = np.reshape(self.samples,(-1,3))
        e3 = time.time()
        print("time:",(e3-s3),"s")
        

        # pbar = tqdm(total = len(result_list))
        # while result_list:
        #     pbar.update(1)
        #     ray_get = ray.get(result_list[0])
        #     if not ray_get:
        #         del result_list[0]
        #         continue
        #     for bi in range(ray_batch):
        #         if not ray_get[bi]:
        #             continue

        #         img = ray_get[bi][0]
        #         target = ray_get[bi][1]
        #         ft_img = ray_get[bi][2]
        #         count[int(target)]+=1
                
        #         self.samples.append(img)
        #         self.targets.append(target)
        #         self.samples_FT.append(ft_img)

        #     del result_list[0]


        # for ni in tqdm(range(len(result_list))):
        #     ray_get = ray.get(result_list[ni])
        #     #ray_get = result_list[ni]
        #     if not ray_get:
        #             continue

        #     #auto_garbage_collect()
        #     for bi in range(ray_batch):
        #         if not ray_get[bi]:
        #             continue
        #         img = ray_get[bi][0]
        #         target = ray_get[bi][1]
        #         #ft_img = ray_get[bi][2]
        #         count[int(target)]+=1
                
        #         self.samples.append(img)
        #         self.targets.append(target)
        #         #self.samples_FT.append(ft_img)


        print("total:",len(self.samples))
        print("DatasetFolderFT_txt init")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index][0]
        target = self.samples[index][1]
        ft_sample = self.samples[index][2]
        # ft_sample = generate_FT(sample)
        # ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        # ft_sample = torch.from_numpy(ft_sample).float()
        # ft_sample = torch.unsqueeze(ft_sample, 0)

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