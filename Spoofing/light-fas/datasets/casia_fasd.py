import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class CasiaFasdDataset(Dataset):
    def __init__(self,root_dir,transform=None,mode='train'):
        self.root_dir = root_dir
        self.transform = transform

        if mode=='all':
            self.csv_path_train = os.path.join(self.root_dir,"casia_crop_list_train.csv")
            self.csv_path_test = os.path.join(self.root_dir,"casia_crop_list_test.csv")
            
            df1 = pd.read_csv(self.csv_path_train,header=None)
            df2 = pd.read_csv(self.csv_path_test,header=None)

            df = pd.concat((df1,df2))
            df = df.reset_index(drop=True)

            self.paths = df[0]
            self.ids = df[1]
            self.labels_str = df[2]

        else:
            self.csv_path = os.path.join(self.root_dir,"casia_crop_list_{}.csv".format(mode))

            df = pd.read_csv(self.csv_path_train,header=None)
            self.paths = df[0]
            self.idx = df[1]
            self.labels_str = df[2]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]
        label_str = self.labels_str[idx]

        cropped_face = cv2.imread(path)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        if label_str=='real':
            label=0
        else:
            label=1

        if self.transform:
            cropped_face = self.transform(label=label, img=cropped_face)['image']
        cropped_face = np.transpose(cropped_face, (2, 0, 1)).astype(np.float32)
        return (torch.tensor(cropped_face), torch.tensor(label, dtype=torch.long))