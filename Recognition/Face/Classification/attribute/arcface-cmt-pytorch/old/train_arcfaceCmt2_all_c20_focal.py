import cv2
import numpy as np
import torch
import os
import pickle
import time

from backbones import get_cmt_model_c20
import random

import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

import h5py
from sklearn.model_selection import train_test_split
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from dataset_agengender_h5 import AgeNGender_Dataset,load_data_hdf5_20_tv_concat,AgeNGender_Dataset_aug,load_data_hdf5_20_tv
from losses import FocalLoss

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torch.nn.functional as F

from torch.optim import lr_scheduler
from collections import OrderedDict

import argparse
import logging
import sys
import copy

from early_stopping import EarlyStopping

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def target_gender(labels,device):
    return torch.tensor(np.where(labels<10,0.0,1.0),dtype=torch.long).to(device)

def target_age(labels,device):
    return F.one_hot(labels.type(torch.int64),num_classes=10).to(device)


def get_args():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--date",type=str,default="20220111")
    parser.add_argument("--prepath",type=str,default="")
    parser.add_argument("--train_data",type=str,nargs='+',default='',help='training dataset path')
    parser.add_argument("--val_data",type=str,nargs='+',default='',help='training dataset path')
    parser.add_argument("--save_name",type=str,default='',help='save_name')
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--train_batch",type=int,default=128)
    parser.add_argument("--val_batch",type=int,default=20)
    parser.add_argument("--image_size",type=int,default=112)
    parser.add_argument("--lr",type=float,default=0.005)
    parser.add_argument("--embedding_size",type=int,default=512)
    parser.add_argument("--num_epoch",type=int,default=100)
    parser.add_argument("--flagAug",type=lambda x: (str(x).lower() == 'true'),default=False)

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

    cfg = get_args()
    train_paths = cfg.train_data
    val_paths = cfg.val_data
    save_name = cfg.save_name
    flagAug = cfg.flagAug

    save_base = "./save_cmt_models_c20_focal/"
    best_base = "./base_cmt_models_c20_focal/"

    makedir(save_base)
    if "nomargin" in train_paths[0]:
        makedir("{}/nomargin-{}/".format(save_base,cfg.date))
        output_path = "{}/nomargin-{}/{}/".format(save_base,cfg.date,save_name)
        final_best_path = "{}/nomargin-{}_{}.pth".format(best_base,cfg.date,save_name)
    else:
        makedir("{}/{}/".format(save_base,cfg.date))
        output_path = "{}/{}/{}/".format(save_base,cfg.date,save_name)
        final_best_path = "{}/{}_{}.pth".format(save_base,cfg.date,save_name)
    
    makedir(output_path)
    makedir(best_base)

    init_logging(output_path)

    makedir(output_path+"/epoch10/")
    makedir(output_path+"/best/")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)


    backbone = get_cmt_model_c20(pretrained_path = "", num_features=cfg.embedding_size)

    load_weight = torch.load(cfg.prepath)

    if type(load_weight)==OrderedDict:
        try:
            backbone.load_state_dict(load_weight)
        except:
            new_state_dict = OrderedDict()
            for n, v in load_weight.items():
                name = n.replace("module.","") 
                new_state_dict[name] = v
            backbone.load_state_dict(new_state_dict)
    else:
        try:
            backbone.load_state_dict(load_weight.module.state_dict())
        except:
            backbone.load_state_dict(load_weight.state_dict())

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

    opt_backbone = torch.optim.SGD(
        params= backbone.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum)
    

    #exp_lr_scheduler = lr_scheduler.StepLR(opt_backbone, step_size=7, gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt_backbone, 'min',factor=0.5)


    if len(train_paths)>1:
        x_train,y_train = load_data_hdf5_20_tv_concat(train_paths)
        x_val, y_val = load_data_hdf5_20_tv_concat(val_paths)
    else:
        x_train,y_train = load_data_hdf5_20_tv(train_paths[0])
        x_val, y_val = load_data_hdf5_20_tv(val_paths[0])


    if flagAug:
        print("Augmentation")
        train_set = AgeNGender_Dataset_aug(x_train,y_train,cfg.image_size,'train_add')
        val_set = AgeNGender_Dataset_aug(x_val,y_val,cfg.image_size,'val')
    else:
        train_set = AgeNGender_Dataset(x_train,y_train,cfg.image_size)
        val_set = AgeNGender_Dataset(x_val,y_val,cfg.image_size)

    print("all train num:",train_set.num_images," val num:",val_set.num_images)

    train_loader = DataLoader(train_set, batch_size=cfg.train_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.val_batch, shuffle=True)

    # for losses
    y_train_female = y_train[np.where(y_train<10)]
    y_train_male = y_train[np.where(y_train>=10)]
    y_val_female = y_val[np.where(y_val<10)]
    y_val_male = y_val[np.where(y_val>=10)]

    cnum_train_female = [len(y_train_female[np.where(y_train_female==x)]) for x in range(10)]
    cnum_train_male = [len(y_train_male[np.where(y_train_male==x)]) for x in range(10,20)]
    cnum_train_total = [ cnum_train_female[i]+cnum_train_male[i] for i in range(10)]

    cnum_val_female = [len(y_val_female[np.where(y_val_female==x)]) for x in range(10)]
    cnum_val_male = [len(y_val_male[np.where(y_val_male==x)]) for x in range(10,20)]
    cnum_val_total = [ cnum_val_female[i]+cnum_val_male[i] for i in range(10)]

    # loss gender
    celoss = nn.CrossEntropyLoss()
    # losses age
    focal_train_female = FocalLoss(classes_num=cnum_train_female)
    focal_train_male = FocalLoss(classes_num=cnum_train_male)   
    focal_train_total = FocalLoss(classes_num=cnum_train_total)

    focal_val_female = FocalLoss(classes_num=cnum_val_female)
    focal_val_male = FocalLoss(classes_num=cnum_val_male)   
    focal_val_total = FocalLoss(classes_num=cnum_val_total)


    num_image = train_set.num_images
    val_num = val_set.num_images

    total_batch_size = cfg.train_batch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    start_epoch = 0
    global_step = 0
    best_loss= float("inf")

    early_stopping = EarlyStopping(patience = 10,verbose=True)

    for epoch in range(start_epoch, cfg.num_epoch):
        backbone.train()
        logging.info('train mode ... ')
        train_loss=0.0
        print("Epoch:",epoch)
        step=0
        for img, label in tqdm(train_loader):

            global_step += 1
            
            label = label.float()
            age_label = torch.tensor([x if x<10 else x-10 for x in label])
            t_age = target_age(age_label,device)
            img = img.to(device)

            opt_backbone.zero_grad()

            g_feature, a_feature = backbone(img)

            a_female = a_feature[:,:10]
            a_male = a_feature[:,10:]
            
            loss_g = celoss(g_feature,target_gender(label,device))
            loss_a_female = focal_train_female(a_female.type(torch.float64),t_age)
            loss_a_male = focal_train_male(a_male.type(torch.float64),t_age)
            
            p_female = torch.reshape(g_feature[:,0],(img.size(0),1))*a_female
            p_male = torch.reshape(g_feature[:,1],(img.size(0),1))*a_male
            p_age = p_female+p_male
            loss_age = focal_train_total(p_age.type(torch.float64),t_age)

            # Ltotal = λA · (LA + LA|G=0 + LA|G=1 ) + λG · LG   ( λA = 0.1, λG = 1 Best)
            #loss_total = 0.5*(loss_age+loss_a_female+loss_a_male)+ 1*loss_g
            loss_total = loss_age+loss_a_female+loss_a_male+loss_g

            loss_total.backward()
            opt_backbone.step()

            #exp_lr_scheduler.step()
            
            train_loss += loss_total.item()

        epoch_train_loss = train_loss/len(train_loader)
        logging.info("Epoch: {} / train Loss: {:.4f}".format(epoch,epoch_train_loss))

        backbone.eval()

        logging.info('val mode ... ')
 

        with torch.no_grad():
            accuracy_age=0
            accuracy_gender=0
            val_loss=0.0
            for val_step,(val_data, val_lb) in enumerate(val_loader):
                
                val_lb = val_lb.float()

                val_age_label = torch.tensor([x if x<10 else x-10 for x in val_lb])
                val_age = target_age(val_age_label,device)

                val_gender = target_gender(val_lb,device)
                val_data = val_data.to(device)
                

                val_g_feature, val_a_feature = backbone.forward(val_data)
                val_a_female = val_a_feature[:,:10]
                val_a_male = val_a_feature[:,10:]


                val_loss_g = celoss(val_g_feature,val_gender)
                val_loss_a_female = focal_val_female(val_a_female.type(torch.float64),val_age)
                val_loss_a_male = focal_val_male(val_a_male.type(torch.float64),val_age)

                val_p_female = torch.reshape(val_g_feature[:,0],(val_data.size(0),1))*val_a_female
                val_p_male = torch.reshape(val_g_feature[:,1],(val_data.size(0),1))*val_a_male
                val_p_age = val_p_female + val_p_male
                val_loss_age = focal_val_total(val_p_age.type(torch.float64),val_age)

                #batch_loss = 0.5*(val_loss_age+val_loss_a_female+val_loss_a_male)+ 1*val_loss_g
                batch_loss = val_loss_age+val_loss_a_female+val_loss_a_male+val_loss_g

                val_loss += batch_loss.item()
                
                # age
                top_p_age, top_class_age = val_p_age.topk(1,dim=1)
                equals_age = top_class_age == val_age.view(*top_class_age.shape)
                accuracy_age += torch.mean(equals_age.type(torch.FloatTensor)).item()
                # gender
                top_p_gender, top_class_gender = val_g_feature.topk(1,dim=1)
                equals_gender = top_class_gender == val_gender.view(*top_class_gender.shape)
                accuracy_gender += torch.mean(equals_gender.type(torch.FloatTensor)).item()


            total_acc_age = accuracy_age/len(val_loader)
            total_acc_gender = accuracy_gender/len(val_loader)

            val_loss = val_loss/len(val_loader)            


            scheduler.step(val_loss)
            early_stopping(val_loss)

            #print("epoch:",epoch,"val total loss:%.4f"%val_loss+" age acc:%.3f"%total_acc_age+" gender acc:%.3f"%total_acc_gender+"\n")
            logging.info("Epoch: {} / val Loss: {:.4f} / val age Acc: {:.4f} / val gender Acc: {:.4f}".format(epoch,val_loss,total_acc_age,total_acc_gender))

        save_epoch10 = output_path+"/epoch10/epoch_"+str(epoch)+"_step_"+str(global_step)+"_val_loss_%.4f"%val_loss+"_age_acc_%.4f"%total_acc_age+"_gender_acc_%.4f"%total_acc_gender+"_backbone.pth"
        save_best = output_path+"/best/epoch_"+str(epoch)+"_step_"+str(global_step)+"_val_loss_%.4f"%val_loss+"_age_acc_%.4f"%total_acc_age+"_gender_acc_%.4f"%total_acc_gender+"_best.pth"

        if epoch%10==0:
            torch.save(backbone.state_dict(),save_epoch10)
        if best_loss>val_loss:
            best_loss = val_loss
            torch.save(backbone.state_dict(),save_best)
            best_model_wts = copy.deepcopy(backbone.state_dict())
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    logging.info("Finish.")

    backbone.load_state_dict(best_model_wts)

    return backbone, final_best_path


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    best_model,final_best_path = main()
    torch.save(best_model,final_best_path)




