import os
import argparse

from PIL import Image

import torch
from torchvision import transforms

from networks.dan import DAN

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Image file or directory for evaluation.')
    parser.add_argument('--model', type=str,default="checkpoints_211101/rafdb_epoch32_acc0.8312_bacc0.8301.pth", help='model path.')
 
    return parser.parse_args()

class Model():
    def __init__(self,model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ["happy","embarrassed","anger","anxious","hurt","sorrow","neutrality",]

        self.model = DAN(num_head=4, num_class=7)
        checkpoint = torch.load(model_path,
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()
    
    def fit(self, path):

        # input is file
        if not os.path.isdir(path) and not ".txt" in path:
            gt = path.split("_")[0]
            img = Image.open(path).convert('RGB')
            img = self.data_transforms(img)
            img = img.view(1,3,224,224)
            img = img.to(self.device)

            with torch.set_grad_enabled(False):
                out, _, _ = self.model(img)
                _, pred = torch.max(out,1)
                index = int(pred)
                label = self.labels[index]

                print('gt:{0}, predict: {1}'.format(gt,label))

        # input is image txt 
        elif ".txt" in path:
            lines = open(path,'r',encoding='utf-8').readlines()

            total = len(lines)
            true_cnt = 0
            total_each = np.zeros(len(self.labels))
            true_each = np.zeros(len(self.labels))

            check_dict = {}

            for k in self.labels:
                check_dict[k]=np.zeros(len(self.labels))

            for line in lines:
                sp = line.strip().split(",")
                _,img_path,gender,age,gt = sp

                total_each[self.labels.index(gt)]+=1

                img = Image.open(img_path).convert('RGB')
                img = self.data_transforms(img)
                img = img.view(1,3,224,224)
                img = img.to(self.device)

                with torch.set_grad_enabled(False):
                    out, _, _ = self.model(img)
                    _, pred = torch.max(out,1)
                    index = int(pred)
                    label = self.labels[index]

                if gt==label:
                    true_cnt+=1
                    true_each[self.labels.index(gt)]+=1

                check_dict[gt][self.labels.index(label)]+=1

                print('gt:{0}, predict: {1}'.format(gt,label))

            print("total:",total," true_cnt:",true_cnt)
            print("total_each:",total_each)
            print("true_each:",true_each)

            print("check_dict")
            for k in check_dict.keys():
                print(k,":",check_dict[k])

            acc = true_cnt/total
            acc_each = true_each/total_each

            print("acc:",acc)
            print("acc_each:",acc_each)



        #input is image directory
        else:
            img_list = [x for x in os.listdir(path) if not "DS" in x]
            total = len(img_list)
            true_cnt = 0
            total_each = np.zeros(len(self.labels))
            true_each = np.zeros(len(self.labels))

            check_dict = {}

            for k in self.labels:
                check_dict[k]=np.zeros(len(self.labels))

            for img_path in img_list:
                if "DS" in img_path: continue
                gt = img_path.split("_")[0]
                total_each[self.labels.index(gt)]+=1

                img = Image.open(path+img_path).convert('RGB')
                img = self.data_transforms(img)
                img = img.view(1,3,224,224)
                img = img.to(self.device)

                with torch.set_grad_enabled(False):
                    out, _, _ = self.model(img)
                    _, pred = torch.max(out,1)
                    index = int(pred)
                    label = self.labels[index]

                if gt==label:
                    true_cnt+=1
                    true_each[self.labels.index(gt)]+=1

                check_dict[gt][self.labels.index(label)]+=1

                print('gt:{0}, predict: {1}'.format(gt,label))

            print("total:",total," true_cnt:",true_cnt)
            print("total_each:",total_each)
            print("true_each:",true_each)

            print("check_dict")
            for k in check_dict.keys():
                print(k,":",check_dict[k])

            acc = true_cnt/total
            acc_each = true_each/total_each

            print("acc:",acc)
            print("acc_each:",acc_each)

            
                



if __name__ == "__main__":
    args = parse_args()

    model = Model(args.model)

    image = args.image
    assert os.path.exists(image)

    model.fit(image)

    
