import os

import PIL.Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from .meta import DEVICE_INFOS

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)




class FaceDataset(Dataset):
    
    def __init__(self, dataset_name, df, split='train', label=None, transform=None, scale_up=1.1, scale_down=1.0, map_size=32, UUID=-1):
        # self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        self.split = split
        self.df = df # img_path,id,label(real or fake)
        self.img_paths = df[0]
        self.id_list = df[1]
        self.label_list = df[2]
        
        self.dataset_name = dataset_name
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.map_size = map_size
        self.UUID = UUID
        print("split:",split)
        print("UUID:",UUID)
        print("dataset:",dataset_name)
        self.face_width = 400

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # video_name = str(self.landmarks_frame.iloc[idx, 1])
        # spoofing_label = self.landmarks_frame.iloc[idx, 0]
        img_path = self.img_paths[idx]
        # label_list[0]=='real' 이면 spoofing_label=1
        spoofing_label = int('real'==self.label_list[idx])

        device_tag = 'real' if spoofing_label else 'fake'

        client_id = self.id_list[idx]

        if self.split == 'train':
            image_x = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB) # rgb image
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = self.transform(PIL.Image.fromarray(image_x))
        else:
            image_x = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB) # rgb image
            image_x_view1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_view2 = image_x_view1
            # import matplotlib.pyplot as plt
            # plt.imshow(image_x);plt.show()

        sample = {"image_x_v1": np.array(image_x_view1),
                  "image_x_v2": np.array(image_x_view2),
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'device_tag': device_tag,
                  'client_id': client_id}
        return sample

    def get_single_image_x(self, img_path):
        image = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB) # rgb image
        h_img, w_img = image.shape[0], image.shape[1]
        h_div_w = h_img / w_img
        image_x = cv2.resize(image, (self.face_width, int(h_div_w * self.face_width)))
        return image_x
