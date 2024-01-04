import os
import warnings
from tqdm import tqdm
import argparse

from PIL import Image,ImageOps
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms


from sklearn.metrics import balanced_accuracy_score

from networks.dan import DAN

import random
import h5py

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_format', type=str, default='./aihub_dataset/FaceEmotion_221031/{}_aligned112.h5')
    parser.add_argument('--save_base',type=str,default="./checkpoints_aihub_230403_newlb/")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--input_size',type=int,default=224)
    parser.add_argument('--num_classes',type=int,default=7)

    return parser.parse_args()


class AihubDataset(data.Dataset):
    def __init__(self, data_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path

        label_dict = {'기쁨':0,'당황':1,'분노':2,'불안':3,'상처':4,'슬픔':5,'중립':6}

        if phase=='train':
            data_path = (self.data_path).format('Training')
        else:
            data_path = (self.data_path).format('Validation')

        df = pd.read_csv(data_path,header=None)

        #self.file_paths=[]
        self.images=[]
        self.label=[]
        
        for val in df.values:
            ori_id,new_id,gender,age,emotion_ko,bboxes_str,ori_path,new_path = val
            image_path = new_path.replace("/data/notebook/shared/Download/","/data/shared/aihub_dataset/")
            lb = label_dict[emotion_ko]
            #self.file_paths.append(image_path)
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            image = image.convert('RGB')
            self.images.append(image)
            self.label.append(lb)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # path = self.file_paths[idx]
        # image = Image.open(path)
        # image = ImageOps.exif_transpose(image)
        # image = image.convert('RGB')
        image = self.images[idx]
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class AihubDatasetH5(data.Dataset):
    def __init__(self, data_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path

        label_dict = {'기쁨':0,'당황':1,'분노':2,'불안':3,'상처':4,'슬픔':5,'중립':6}

        if phase=='train':
            data_path = (self.data_path).format('Training')
        else:
            data_path = (self.data_path).format('Validation')

        hf = h5py.File(data_path,'r')

        self.images = hf['image']
        self.labels = hf['label']


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)
        
        return image, label

class AihubDatasetH5_newlb(data.Dataset):
    def __init__(self, data_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path

        self.label_rematch = {0:0, 1:1, 2:2, 3:1, 4:3, 5:3, 6:4}

        if phase=='train':
            data_path = (self.data_path).format('Training')
        else:
            data_path = (self.data_path).format('Validation')

        hf = h5py.File(data_path,'r')

        self.images = hf['image']
        self.labels = hf['label']


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label_ori = self.labels[idx]
        label = self.label_rematch[label_ori]

        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)
        
        return image, label

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1+num_head/var)
        else:
            loss = 0
            
        return loss

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    input_size=args.input_size

    if not os.path.exists(args.save_base):
        os.mkdir(args.save_base)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=args.num_head,num_class=args.num_classes)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(input_size, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    #train_dataset = AihubDatasetH5(args.data_format, phase = 'train', transform = data_transforms)    
    train_dataset = AihubDatasetH5_newlb(args.data_format, phase = 'train', transform = data_transforms)
    
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

    #val_dataset = AihubDatasetH5(args.data_format, phase = 'test', transform = data_transforms_val)   
    val_dataset = AihubDatasetH5_newlb(args.data_format, phase = 'test', transform = data_transforms_val) 

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_af = AffinityLoss(device)
    criterion_pt = PartitionLoss()

    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)

            loss = criterion_cls(out,targets) + 1* criterion_af(feat,targets) + 1*criterion_pt(heads)  #89.3 89.4

            loss.backward()
            optimizer.step()
            

            #print("train iter:",iter_cnt," Loss:",loss.item())

            running_loss += loss.item()
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []

            model.eval()
            for (imgs, targets) in tqdm(val_loader):
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                out,feat,heads = model(imgs)
                loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

                running_loss += loss.item()
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)


                #print("val iter:",iter_cnt," Loss:",loss.item())
                
                baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
            running_loss = running_loss/iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)

            bacc = np.around(np.mean(baccs),4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, bacc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))


            if acc == best_acc:
        
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join(args.save_base, "aihub_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(bacc)+".pth"))
                tqdm.write('Model saved.')

        
if __name__ == "__main__":        
    run_training()