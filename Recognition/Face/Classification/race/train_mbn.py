import cv2
import numpy as np
import torch
import os
import pickle
import time

from backbones.get_models import get_model
import random

import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

import h5py
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from dataset import DatasetLoader, DatasetLoader_nottrans

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torch.nn.functional as F

from torch.optim import lr_scheduler

import argparse
import logging
import sys
import copy

from utils.early_stopping import EarlyStopping
from utils.utils_config import get_config

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def get_args():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",type=str,default="configs/race.py")

    args = parser.parse_args()
    return args

def init_logging(models_root):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s - %(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)


def main():
    opts = get_args()
    cfg = get_config(opts.config)

    data_prefix = cfg.prefix
    train_path = cfg.train_data
    val_path = cfg.val_data
    output = cfg.output
    

    makedir(output)
    init_logging(output)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    if cfg.loss=='celoss':
        criterion = nn.CrossEntropyLoss()
    else:
        print("loss error")
        exit()

    backbone = get_model(cfg.network,num_classes=cfg.num_classes,load_type=cfg.load_type, \
                            input_size=cfg.image_size,pretrained=cfg.pretrained)
    
    # requires_grad parameter print
    print("reauires_grad=True parameters")
    for name, param in backbone.named_parameters():
        if param.requires_grad:
            print(name)
    print()
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    ## multi-gpu
    if torch.cuda.device_count()>1:
        backbone = nn.DataParallel(backbone)
    backbone = backbone.to(device)

    best_model_wts = copy.deepcopy(backbone.state_dict())

    if cfg.optim_way=='sgd':
        opt_backbone = torch.optim.SGD(
            params= backbone.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum)
    elif cfg.optim_way=='adam':
        opt_backbone = torch.optim.Adam(backbone.parameters(),lr=cfg.lr)

    
    #exp_lr_scheduler = lr_scheduler.StepLR(opt_backbone, step_size=7, gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt_backbone, 'min',factor=0.5)
    #scheduler = lr_scheduler.ReduceLROnPlateau(opt_backbone, 'max',factor=0.5)
    

    ## dataset load
    if cfg.trans==True:
        train_set = DatasetLoader(data_prefix,train_path,cfg.image_size)
    else:
        train_set = DatasetLoader_nottrans(data_prefix,train_path,cfg.image_size)
    val_set = DatasetLoader_nottrans(data_prefix,val_path,cfg.image_size)

    print("all train num:",train_set.num_images," val num:",val_set.num_images)

    train_loader = DataLoader(train_set, batch_size=cfg.train_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.val_batch, shuffle=True)
    ## 

    num_image = train_set.num_images
    val_num = val_set.num_images

    total_batch_size = cfg.train_batch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    start_epoch = 0
    global_step = 0
    best_loss= float("inf")
    best_acc = float("-inf")

    early_stopping = EarlyStopping(patience = 10,verbose=True)

    for epoch in range(start_epoch, cfg.num_epoch):
        backbone.train()
        logging.info('train mode ... ')
        train_loss=0.0
        print("Epoch:",epoch)
        step=0
        for img, label in tqdm(train_loader):

            global_step += 1
            
            img,label = img.cuda(),label.cuda()
        
            opt_backbone.zero_grad()

            fc = backbone(img)

            loss = criterion(fc,label)

            loss.backward()
            opt_backbone.step()

            #exp_lr_scheduler.step()
            train_loss += loss.item()

        epoch_train_loss = train_loss/len(train_loader)
        logging.info("Epoch: {} / train Loss: {:.4f}".format(epoch,epoch_train_loss))

        backbone.eval()

        logging.info('val mode ... ')
 

        with torch.no_grad():
            accuracy=0
            val_loss=0.0
            for val_step,(val_data, val_lb) in enumerate(val_loader):
                
                val_data,val_lb = val_data.cuda(),val_lb.cuda()
                
                val_fc = backbone(val_data)

                batch_loss = criterion(val_fc,val_lb)
                val_loss += batch_loss.item()
                
                top_p, top_class = val_fc.topk(1,dim=1)
                equals = top_class == val_lb.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
               

            total_acc = accuracy/len(val_loader)
            val_loss = val_loss/len(val_loader)            

            scheduler.step(val_loss)
            #scheduler.step(accuracy)
            early_stopping(val_loss)

            logging.info("Epoch: {} / val Loss: {:.4f} / val Acc: {:.4f}".format(epoch,val_loss,total_acc))

        save_epoch = os.path.join(output,"epoch_{}.pth".format(epoch))
        save_best = os.path.join(output,"best.pth")

        if epoch%10==0:
            torch.save(backbone.state_dict(),save_epoch)
        if best_loss>val_loss:
            best_loss = val_loss
            torch.save(backbone.state_dict(),save_best)
            best_model_wts = copy.deepcopy(backbone.state_dict())
        # if best_acc<total_acc:
        #     best_acc = total_acc
        #     torch.save(backbone.state_dict(),save_best)
        #     best_model_wts = copy.deepcopy(backbone.state_dict())

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    logging.info("Finish.")

    backbone.load_state_dict(best_model_wts)
    final_best_path = os.path.join(output,"final_best.pth")
    return backbone, final_best_path


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    best_model,final_best_path = main()
    torch.save(best_model,final_best_path)





