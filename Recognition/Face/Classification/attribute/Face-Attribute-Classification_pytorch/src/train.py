import os
import numpy as np
import torch
import argparse
import logging
import sys
from tqdm import tqdm

from get_config import get_config
from model import build_model
from datasets import CelebALoader
from early_stopping import EarlyStopping
from evaluate import eval

import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Dataset
from torch import nn

def get_args():
    parser = argparse.ArgumentParser(description='Train Attribute Models')
    parser.add_argument('--cfg_path', type=str,
        default='configs/efficientNet_B0_celebA.py')

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    args = get_args()
    cfg_path = args.cfg_path
    cfg = get_config(cfg_path)

    # for save
    output = cfg.output
    if not os.path.exists(output):
        os.makedirs(output)
    
    init_logging(output)

    # get model
    model = build_model(cfg.network,cfg.num_classes,cfg.pretrain_backbone,True)
    print("reauires_grad=True parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    ## multi-gpu
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # optimizer n LR
    optimizer = torch.optim.Adam(model.parameters(),cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=6,min_lr=1e-13)

    # Loss
    criterion = BCELoss()

    # Dataset
    train_set = CelebALoader(cfg.dataset_format.format('train'),cfg.input_size)
    val_set = CelebALoader(cfg.dataset_format.format('val'),cfg.input_size)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True)

    # early
    early_stopping = EarlyStopping(patience = 10,verbose=True)

    best_loss= float("inf")
    best_acc = float("-inf")

    for epoch in range(cfg.epochs):
        print("Epoch:",epoch)

        train_loss=0.0
        logging.info("Train...")
        for img,labels in tqdm(train_loader):
            img,labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            out = model(img)

            loss = criterion(out.to(torch.float32),labels.to(torch.float32))
            train_loss+=loss.item()

            loss.backward()
            optimizer.step()

        epoch_train_loss = train_loss/len(train_loader)
        logging.info("Epoch: {} / train Loss: {:.4f}".format(epoch,epoch_train_loss))

        model.eval()
        logging.info("Validation...")
        val_loss=0.0
        val_acc=0.0
        val_f1 = 0.0
        val_prec=0.0
        val_rec=0.0
        val_auc=0.0

        with torch.no_grad():
            for img,labels in tqdm(val_loader):
                img,labels = img.cuda(), labels
                out = model(img)
                out = out.cpu()

                loss, acc, f1_score, precision, recall, auc = eval(out,labels,criterion)
                val_loss+=loss
                val_acc+=acc
                val_f1+=f1_score
                val_prec+=precision
                val_rec+=recall
                val_auc+=auc

            val_loss = val_loss/len(val_loader)
            val_acc = val_acc/len(val_loader)
            val_f1 = val_f1/len(val_loader)
            val_prec = val_prec/len(val_loader)
            val_rec = val_rec/len(val_loader)
            val_auc = val_auc/len(val_loader)

            scheduler.step(val_loss)
            early_stopping(val_loss)

            logging.info("Epoch: {}/ val Loss: {}/ val Acc: {}".format(epoch,val_loss,val_acc))
            logging.info("f1-score: {}/ precision: {}/ recall: {}/ acu: {}".format(val_f1,val_prec,val_rec,val_auc))


        save_epoch = os.path.join(output,"epoch_{}.pth".format(epoch))
        save_best = os.path.join(output,"best_epoch{}.pth".format(epoch))

        if epoch%5==0:
            torch.save(model.state_dict(),save_epoch)

        if best_loss>val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),save_best)


        if early_stopping.early_stop:
            print("Early stopping")
            break

    logging.info("Finish.")



if __name__ == "__main__":
    main()