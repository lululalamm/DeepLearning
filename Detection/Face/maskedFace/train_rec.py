import os
import numpy as np
import argparse
from tqdm import tqdm
import logging
import sys
import time

from torch import nn
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import MaskDatasetrec
from backbone.get_models import get_model
from losses.focal_loss import FocalLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default='mnasnet075',help="mnasnet075 / mobilenetv3-small / mobilenetv3-large / mobilefacenet")
    parser.add_argument("--rec_path",type=str,default="/data/notebook/shared/temp~~")
    parser.add_argument("--save_base",type=str,default='./save_models/')
    parser.add_argument("--save_name",type=str,default='train')
    parser.add_argument("--train_batch_size",type=int,default=512)
    parser.add_argument("--val_batch_size",type=int,default=512)
    parser.add_argument("--lr",type=float,default=0.0005)
    parser.add_argument("--num_classes",type=int,default=2)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--loss_metric",type=str,default='celoss',help='celoss or focalloss or bceloss')
    parser.add_argument("--data_aug",action='store_true',default=False)
    parser.add_argument("--rec_version",type=str,default='')
    parser.add_argument("--load_type",type=int,default=1)
    parser.add_argument("--input_size",type=int,default=112)
    args = parser.parse_args()
    return args

def init_logging(models_root,args):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s - %(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)

    logging.info("model: {}".format(args.model_name))

    logging.info("rec_path: {}".format(args.rec_path))
    logging.info("rec_version: {}".format(args.rec_version))
    logging.info("augmentation: {}".format(args.data_aug))

    logging.info("img_size: {}".format(args.input_size))

    logging.info("load_type: {}".format(args.load_type))
    logging.info("save_path: {}".format(models_root))
    logging.info("train_batch: {}".format(args.train_batch_size))
    logging.info("val_batch: {}".format(args.val_batch_size))
    logging.info("lr: {}".format(args.lr))
    logging.info("num_classes: {}".format(args.num_classes))
    logging.info("epochs: {}".format(args.epochs))
    logging.info("loss: {}".format(args.loss_metric))



def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    # save_path
    if not os.path.exists(os.path.join(args.save_base,args.save_name)):
        os.makedirs(os.path.join(args.save_base,args.save_name))

    summary_base = os.path.join(args.save_base,args.save_name,"tensorboard")
    if not os.path.exists(summary_base):
        os.mkdir(summary_base)

    save_base = os.path.join(args.save_base,args.save_name)
    init_logging(save_base,args)

    # loss , optim , scheduler
    if args.loss_metric=='celoss':
        criterion = nn.CrossEntropyLoss()
        model_class=args.num_classes
    elif args.loss_metric=='focal_loss':
        criterion = FocalLoss()
        model_class=args.num_classes
    elif args.loss_metric=='bceloss':
        criterion = nn.BCEWithLogitsLoss()
        model_class=1
    else:
        print("loss metric Error,,,", args.loss_metric)
        exit()


    # model load
    model = get_model(args.model_name,num_classes=model_class,load_type=args.load_type,input_size=args.input_size)
    
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model = model.to(device)


    optimizer = torch.optim.SGD(
        params= model.parameters(),
        lr=args.lr,
        momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1)

    rec_version = args.rec_version
    if not rec_version:
        train_name = 'train'
        val_name = 'val'
    else:
        train_name = 'train{}'.format(rec_version)
        val_name = 'val{}'.format(rec_version)

    # data loader
    trainset = MaskDatasetrec(args.rec_path,train_name,aug=args.data_aug,input_size=args.input_size)
    valset = MaskDatasetrec(args.rec_path,val_name,aug=args.data_aug,input_size=args.input_size)

    train_loader = DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0)

    val_loader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=0)


    # train
    best_loss = float("inf")
    best_acc = float("-inf")

    writer = SummaryWriter(summary_base)

    logging.info("Start Training....")
    logging.info("Tensorboard log dir: {}".format(summary_base))

    summary_num = 10
    total_step=0

    running_train_loss = 0.

    for epoch in range(args.epochs):
        train_loss=0.0
        model.train()
        load_start = time.time()
        for image, target in tqdm(train_loader):
            image = image.to(device)
            target = target.to(device)
            load_end = time.time()

            optimizer.zero_grad()

            infer_start = time.time()
            output = model(image)
            infer_end = time.time()

            loss_start=time.time()
            loss = criterion(output,target)

            loss.backward()
            optimizer.step()

            train_loss+=loss.item()
            running_train_loss += loss.item()
            loss_end = time.time()

            sum_start = time.time()
            if total_step%summary_num==0 and total_step!=0:
                loss_board = running_train_loss / summary_num
                writer.add_scalar(
                        'Training/Loss_per_step', loss_board, total_step)

                running_train_loss = 0.

            sum_end = time.time()

            # print("load: {} s".format(load_end-load_start))
            # print("infer: {} s".format(infer_end-infer_start))
            # print("loss : {} s".format(loss_end-loss_start))
            # print("sum: {} s".format(sum_end-sum_start))
            # exit()

            total_step += 1
        epoch_train_loss = train_loss/len(train_loader)
        writer.add_scalar(
                        'Training/Loss_per_epoch', epoch_train_loss, epoch)
        logging.info("Epoch: {} / train Loss: {:.4f}".format(epoch,epoch_train_loss))

        model.eval()

        with torch.no_grad():
            acc = 0
            val_loss=0.0
            for image, target in tqdm(val_loader):
                image = image.to(device)
                target = target.to(device)

                output = model(image)
                loss = criterion(output,target)

                val_loss += loss.item()

                output = torch.softmax(output,dim=-1)
                top_p, top_index = output.topk(1,dim=1)
                equals = top_index == target.view(*top_index.shape)
                acc += torch.mean(equals.type(torch.FloatTensor)).item()

            val_acc = acc/len(val_loader)
            val_loss = val_loss/len(val_loader)

            scheduler.step(val_loss)
            logging.info("Epoch: {} / val Loss: {:.4f} / val Acc: {:.4f}".format(epoch,val_loss,val_acc))

            writer.add_scalar(
                        'Validation/Loss_per_epoch', val_loss, epoch)
            writer.add_scalar(
                        'Validation/Acc_per_epoch', val_acc, epoch)

        if best_loss>val_loss:
            save_path = os.path.join(save_base,"best_epoch-{}_loss-{:.4f}_acc-{:.4f}.pth".format(epoch,val_loss,val_acc))
            torch.save(model.state_dict(),save_path)
            best_loss = val_loss

        if best_acc<val_acc:
            save_path = os.path.join(save_base,"best_epoch-{}_acc-{:.4f}_loss-{:.4f}.pth".format(epoch,val_acc,val_loss))
            torch.save(model.state_dict(),save_path)
            best_acc = val_acc


    print("Train Finish")
    save_path = os.path.join(save_base,"last.pth")
    torch.save(model.state_dict(),save_path)


















if __name__ == "__main__":
    args = get_args()
    train(args)






