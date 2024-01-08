# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm

import torch
import pdb
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MultiFTNet import MultiFTNet_resnet
from src.data_io.dataset_loader import get_train_loader,get_train_loader_txt,get_train_loader_h5

import time

class TrainMain:
    def __init__(self, conf,txt=False,h5=False,ray=True,slack_api=None,ycrcb=False):
        torch.autograd.set_detect_anomaly(True)

        self.conf = conf
        self.ycrcb = ycrcb
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        if txt:
            self.train_loader = get_train_loader_txt(self.conf,ray,ycrcb=self.ycrcb)
        elif h5:
            self.train_loader = get_train_loader_h5(self.conf,ycrcb=self.ycrcb)
        else:
            self.train_loader = get_train_loader(self.conf,ycrcb=self.ycrcb)

        self.slack_api=slack_api

    def train_model(self):
        
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        self.model.train()
        running_loss = 0.
        running_acc = 0.
        running_loss_cls = 0.
        running_loss_ft = 0.
        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_lr())

            #train_iter = iter(self.train_loader)

            for sample, ft_sample, target in tqdm(self.train_loader):
            #for data in tqdm(train_iter):
                imgs = [sample, ft_sample]
                labels = target

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)

                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss', loss_board, self.step)
                    acc_board = running_acc / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Acc', acc_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar(
                        'Training/Learning_rate', lr, self.step)
                    loss_cls_board = running_loss_cls / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss_cls', loss_cls_board, self.step)
                    loss_ft_board = running_loss_ft / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss_ft', loss_ft_board, self.step)

                    running_loss = 0.
                    running_acc = 0.
                    running_loss_cls = 0.
                    running_loss_ft = 0.

                if self.step == 10:
                    time_stamp = get_time()
                    _ = self._save_state(time_stamp, extra=self.conf.job_name)
                if self.step % self.save_every == 0 and self.step != 0:
                    time_stamp = get_time()
                    _ = self._save_state(time_stamp, extra=self.conf.job_name)

            

            self.schedule_lr.step()

        time_stamp = get_time()
        save_path_last = self._save_state(time_stamp, extra=self.conf.job_name,last=True)
        #self.slack_api.send_message("last model upload")
        #self.slack_api.upload_file(save_path_last)
        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device,dtype=torch.int64)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        # print("image")
        # print(imgs[0])
        try:
            loss.backward()
        except Exception as e:
            #self.slack_api.send_message("train error:",e)
            print("Error loss backward:",e)
            print("labels:",labels)
            print("loss:",loss)
            print("loss_cls:",loss_cls)
            print("loss_fea:",loss_fea)
            print("embeddings")
            print(embeddings)
            print("feature map")
            print(feature_map)
            print("img0")
            print(imgs[0])
            print("img1")
            print(imgs[1])

            print("gradient print")
            for name, param in self.model.module.named_parameters():
                #print(name, torch.isnan(param.grad))
                if torch.isnan(param.grad).any():
                    print(name,": nan gradient found")
                    print(torch.isnan(param.grad))

            #pdb.set_trace()
            exit()
            #pdb.set_trace()
        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 5.)
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _define_network(self):

        if self.conf.network=='nasnet':
            print("Load")
            param = {
                'num_classes': self.conf.num_classes,
                'img_channel': self.conf.input_channel,
                'embedding_size': self.conf.embedding_size,
                'conv6_kernel': self.conf.kernel_size}

            model = MultiFTNet(**param).to(self.conf.device)

        elif self.conf.network=='resnet':
            param = {
                'num_classes':self.conf.num_classes,
                'embedding_size':self.conf.embedding_size,
                'network':self.conf.resnet_name,
                'fp16':True
            }
            model = MultiFTNet_resnet(**param).to(self.conf.device)

        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None,last=False):
        save_path = self.conf.model_path
        if last:
            save_path_all = save_path + "/last.pth"
            torch.save(self.model.state_dict(),save_path_all)
        else:
            torch.save(self.model.state_dict(), save_path + '/' +
                    ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))
            save_path_all = save_path + '/' +('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step))
        return save_path_all
