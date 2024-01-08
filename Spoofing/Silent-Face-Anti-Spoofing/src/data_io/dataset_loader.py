# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

import torch
from torch.utils.data import Sampler, BatchSampler, DataLoader ,RandomSampler
from src.data_io.dataset_folder import DatasetFolderFT,DatasetFolderFT_txt,DatasetFolderFT_h5, DatasetFolderFT_h5_sampler
from src.data_io.dataset_folder_ray import DatasetFolderFT_txt_RAY
from src.data_io import transform as trans
from torchvision import transforms as T


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)

def get_train_loader_txt(conf,ray=True,ycrcb=False):

    if ycrcb:
        train_transform = trans.Compose([
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size),
                                    scale=(0.9, 1.1)),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor()
        ])
    else:
        train_transform = trans.Compose([
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size),
                                    scale=(0.9, 1.1)),
            trans.ColorJitter(brightness=0.4,
                            contrast=0.4, saturation=0.4, hue=0.1),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor()
        ])
    #root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    root_path = conf.train_root_txt_path
    print("root_path txt:",root_path)
    if ray:
        trainset = DatasetFolderFT_txt_RAY(root_path,conf.new_base, train_transform,
                                None, conf.ft_width, conf.ft_height,ycrcb=ycrcb)
    else:
        trainset = DatasetFolderFT_txt(root_path,conf.new_base, train_transform,
                                None, conf.ft_width, conf.ft_height,ycrcb=ycrcb)
    print("train loader start")
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0)
    return train_loader

def get_train_loader_h5(conf,norm_input=False,ycrcb=False):
    # train_transform = trans.Compose([
    #     trans.ToPILImage(),
    #     trans.RandomResizedCrop(size=tuple(conf.input_size),
    #                             scale=(0.9, 1.1)),
    #     trans.ColorJitter(brightness=0.4,
    #                       contrast=0.4, saturation=0.4, hue=0.1),
    #     trans.RandomRotation(10),
    #     trans.RandomHorizontalFlip(),
    #     trans.ToTensor()
    # ])
    if norm_input:
        if ycrcb:
            train_transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(size=tuple(conf.input_size),
                                        scale=(0.9, 1.1)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

        else:
            train_transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(size=tuple(conf.input_size),
                                        scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.4,
                                contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    else:
        if ycrcb:
            train_transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(size=tuple(conf.input_size),
                                        scale=(0.9, 1.1)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                trans.ToTensor()
                ])
        else:
            train_transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(size=tuple(conf.input_size),
                                        scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.4,
                                contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                trans.ToTensor()
                ])

    # shearing 
    # shear = transforms.RandomAffine(
    #         degrees=0, translate=None,
    #         scale=None, shear=15)

    #root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    root_path = conf.h5_path
    print("root_path h5:",root_path)

    #trainset = DatasetFolderFT_h5_sampler(root_path,train_transform,None)
    trainset = DatasetFolderFT_h5(root_path,train_transform,None,conf.ft_width, conf.ft_height,ycrcb=ycrcb)
    print("train loader start")
    # train_loader = DataLoader(
    #     trainset,
    #     batch_size=conf.batch_size,
    #     sampler=BatchSampler(RandomBatchSampler(trainset, conf.batch_size), batch_size=conf.batch_size, drop_last=False))

    shuffle_flag = True
    print("shuffle:",shuffle_flag)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=shuffle_flag,
        pin_memory=True,
        num_workers=0)

        
    return train_loader

# def get_train_loader_h5(conf):
#     train_transform = trans.Compose([
#         trans.ToPILImage(),
#         trans.RandomResizedCrop(size=tuple(conf.input_size),
#                                 scale=(0.9, 1.1)),
#         trans.ColorJitter(brightness=0.4,
#                           contrast=0.4, saturation=0.4, hue=0.1),
#         trans.RandomRotation(10),
#         trans.RandomHorizontalFlip(),
#         trans.ToTensor()
#     ])
#     #root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
#     root_path = conf.h5_path
#     print("root_path txt:",root_path)

#     trainset = DatasetFolderFT_h5(root_path,train_transform,None)
#     print("train loader start")
#     train_loader = DataLoader(
#         trainset,
#         batch_size=conf.batch_size,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=0)

#     return train_loader

def get_train_loader(conf,ycrcb=False):
    if ycrcb:
        train_transform = trans.Compose([
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size),
                                    scale=(0.9, 1.1)),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor()
        ])
    else:
        train_transform = trans.Compose([
            trans.ToPILImage(),
            trans.RandomResizedCrop(size=tuple(conf.input_size),
                                    scale=(0.9, 1.1)),
            trans.ColorJitter(brightness=0.4,
                            contrast=0.4, saturation=0.4, hue=0.1),
            trans.RandomRotation(10),
            trans.RandomHorizontalFlip(),
            trans.ToTensor()
        ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    print("root_path:",root_path)
    trainset = DatasetFolderFT(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height,ycrcb=ycrcb)
    print("train loader start")
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8)
    return train_loader
