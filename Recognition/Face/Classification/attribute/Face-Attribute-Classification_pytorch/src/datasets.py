import os
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import h5py

class CelebALoader(Dataset):
    def __init__(self,h5_path,input_size=224):
        self.transform = T.Compose([ T.ToPILImage(),
                        T.Resize((input_size, input_size)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
        
        self.hf = h5py.File(h5_path)
        self.images = self.hf['images']
        self.labels = self.hf['labels']
        self.label_list = self.hf['label_list']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        
        image = self.images[idx]
        labels = self.labels[idx]
        
        image = self.transform(image)
        
        return image,labels
        
        