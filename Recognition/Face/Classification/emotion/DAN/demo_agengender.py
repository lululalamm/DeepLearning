import os
import argparse

from PIL import Image

import torch
from torchvision import transforms

from networks.dan import DAN

import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    # test data property_test_filtering_0603_aligned, aligned_all_0205
    parser.add_argument('image', type=str, default = 'aligned_all_0205/' ,help='Image file or directory for evaluation.')
    parser.add_argument('--model', type=str,default="checkpoints_agengender_211108/rafdb_epoch22_acc0.8321_bacc0.7814.pth", help='model path.')
    parser.add_argument('--save_pkl',type=bool,default=True)
 
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
        self.labels = ["1_1","1_2","1_3","1_4","1_5","1_6","1_7","1_8",
                "2_1","2_2","2_3","2_4","2_5","2_6","2_7","2_8"]

        self.model = DAN(num_head=4, num_class=16)
        checkpoint = torch.load(model_path,
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()
    
    def fit(self, path,pkl_path):

        # test data only directory 
        total =0
        true_cnt = 0

        total_each = np.zeros(len(self.labels))
        true_each = np.zeros(len(self.labels))

        check_dict = {}

        for k in self.labels:
            check_dict[k]=np.zeros(len(self.labels))

        if pkl_path:
            sm_list=[]

        for gt in os.listdir(path):
            if "DS" in gt: continue

            for img in os.listdir(path+gt):
                if "DS" in img: continue

                img_path = os.path.join(path,gt,img)
                total+=1
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

                if pkl_path:
                    sm_lb=[]

                    sm_lb.append(out)
                    sm_lb.append(self.labels.index(gt))
                    sm_list.append(sm_lb)

                if gt==label:
                    true_cnt+=1
                    true_each[self.labels.index(gt)]+=1

                check_dict[gt][self.labels.index(label)]+=1

                print('gt:{0}, predict: {1}'.format(gt,label))

        if pkl_path:
            pickle.dump(sm_list,open(pkl_path,'wb'))

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

        if pkl_path:
            print("Save pkl:",pkl_path)





if __name__ == "__main__":
    args = parse_args()

    model = Model(args.model)

    image = args.image
    assert os.path.exists(image)

    if args.save_pkl:
        model_split = args.model.split("/")[0].split("_")[-1]+"_"+args.model.split("/")[-1].split("_")[1]
        image_split = image.split("/")[-2]
        pkl_path = "./save_pkl/"+model_split+"_"+image_split+".pkl"
    else:
        pkl_path = ""


    model.fit(image,pkl_path)

    
