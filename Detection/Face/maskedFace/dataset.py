import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import cv2
from tqdm import tqdm

import h5py
import mxnet as mx


lb_dict={'unmask':0,'mask':1}

class MaskDataset(Dataset):
    def __init__(self,txt_path,image_base):
        
        self.images=[]
        self.targets=[]

        self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        lines = open(txt_path,'r').readlines()
        for line in tqdm(lines):
            image_path = os.path.join(image_base,line.strip())
            target_str = line.split(".jpg")[0].split("_")[-1]
            target = lb_dict[target_str]
            input_image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)

            self.images.append(input_image)
            self.targets.append(target)

    def __getitem__(self,index):
        image = self.images[index]
        target = self.targets[index]

        image = self.transform(image)
        #image = torch.unsqueeze(image,dim=0)

        return image,target

    def __len__(self):
        return len(self.targets)


class MaskDataseth5(Dataset):
    def __init__(self,h5path):

        self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        hf = h5py.File(h5path,'r')
        self.images =  hf['image'] # RGB
        self.targets = hf['target']
        self.idnames = hf['id']

    def __getitem__(self,index):
        image = self.images[index]
        target = self.targets[index]

        image = self.transform(image)
        #image = torch.unsqueeze(image,dim=0)

        return image,target

    def __len__(self):
        return len(self.targets)


class MaskDatasetrec(Dataset):
    def __init__(self,rec_path,tv,aug=False,input_size=112):
        
        self.input_size = input_size
        print("dataset input size:",input_size)
        
        if aug and 'train' in tv:
            self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(self.input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.RandomAffine(degrees=0,translate=None,scale=None,shear=20),
                        transforms.RandomEqualize(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                    ])

        else:
            self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(self.input_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])


        path_rec = os.path.join(rec_path,"{}.rec".format(tv))
        path_idx = os.path.join(rec_path,"{}.idx".format(tv))
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_idx, path_rec, 'r')

        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        self.imgidx = np.array(list(self.imgrec.keys))

        self.tv = tv


    def __getitem__(self,index):

        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        
        image = mx.image.imdecode(img).asnumpy() # rgb
        target = int(label[0]) # label = [mask_lb, id_lb]

        image = self.transform(image)

        return image,target

    def __len__(self):
        return len(self.imgidx)
