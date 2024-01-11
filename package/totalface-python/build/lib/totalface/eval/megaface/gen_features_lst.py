import argparse
import os
import struct

import cv2
import numpy as np
import torch
import tqdm
from tqdm import tqdm
import sys
from collections import OrderedDict
import torchvision.transforms as transforms

from PIL import Image
from ...utils.util_common import torch2numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def walkdir(folder, ext):
    # Walk through each files in a directory
    for dirpath, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.lower().endswith(ext)]:
            yield os.path.abspath(os.path.join(dirpath, filename))

def gen_feature_model(prob_name,model,network,path,lst_path,save_base,img_cuda,batch_size=128):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    

    print('gen features {}...'.format(path))
    if not os.path.exists(os.path.join(save_base,prob_name)):
        os.makedirs(os.path.join(save_base,prob_name))
    # Preprocess the total files count
    files = []
    for line in open(lst_path, 'r'):
        file_path = os.path.join(path,line.strip())
        files.append(file_path)
    file_count = len(files)

    # model infer
    for start_idx in range(0, file_count, batch_size):
        end_idx = min(file_count, start_idx + batch_size)
        length = end_idx - start_idx

        imgs = torch.zeros([length, 3, 112, 112], dtype=torch.float)
        for idx in range(0, length):
            i = start_idx + idx
            filepath = files[i]
            imgs[idx] = get_image(cv2.imread(filepath), transform)
        if img_cuda:
            imgs = imgs.cuda()
        # infer
        features = model.net(imgs)
        if isinstance(features,list) and len(features)==1:
            features = features[0]
        features = torch2numpy(features)

        for idx in range(0, length):
            i = start_idx + idx
            filepath = files[i]
            a = filepath.split("/")[-2]
            b = filepath.split("/")[-1]
            tarfile = os.path.join(save_base,prob_name,a)
            if not os.path.exists(tarfile):
                os.makedirs(tarfile)
            tarfile = os.path.join(tarfile,'{}_{}.bin'.format(b,network))
            feature = features[idx]
            write_feature(tarfile, feature / np.linalg.norm(feature))
    

def get_image(img, transformer):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


def read_feature(filename):
    f = open(filename, 'rb')
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
    mat = np.fromstring(f.read(rows * 4), dtype=np.dtype('float32'))
    return mat.reshape(rows, 1)


def write_feature(filename, m):
    header = struct.pack('iiii', m.shape[0], 1, 4, 5)
    f = open(filename, 'wb')
    f.write(header)
    f.write(m.data)

# if image path megaface_images/sub0/sub1/image_name
#               facescrub_images/sub1/image_name ,
# save format : save_base / prob_name / sub1 / image_name_network.bin
def remove_noise(network,mega_noise, scrub_noise,save_base):
    for line in open(mega_noise, 'r'):
        if line.startswith('#'):
            continue
        a = line.split("/")[-2].strip()
        b = line.split("/")[-1].strip()
        filename = os.path.join(save_base,'megaface',a,'{}_{}.bin'.format(b,network))

    
        if os.path.exists(filename):
            print("remove:",filename)
            os.remove(filename)

    
    for line in open(scrub_noise,'r'):
        if line.startswith('#'):
            continue
        a = "_".join(line.split("_")[:-1]).strip()
        b = line.split("/")[-1].strip()
        filename = os.path.join(save_base,'facescrub',a,'{}_{}.bin'.format(b,network))
        if os.path.exists(filename):
            print("remove:",filename)
            os.remove(filename)
    
        


def test():
    root1 = '/root/lin/data/FaceScrub_aligned/Benicio Del Toro'
    root2 = '/root/lin/data/FaceScrub_aligned/Ben Kingsley'
    for f1 in os.listdir(root1):
        for f2 in os.listdir(root2):
            if f1.lower().endswith('.bin') and f2.lower().endswith('.bin'):
                filename1 = os.path.join(root1, f1)
                filename2 = os.path.join(root2, f2)
                fea1 = read_feature(filename1)
                fea2 = read_feature(filename2)
                print(((fea1 - fea2) ** 2).sum() ** 0.5)


def match_result():
    with open('matches_facescrub_megaface_0_1000000_1.json', 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
        for i in range(len(load_dict)):
            print(load_dict[i]['probes'])




def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--mega_dir', default='', help='megaface images directory')
    parser.add_argument('--scrub_dir', default='', help='megaface facescrub images directory')
    parser.add_argument('--network',default='r18',help='model network')
    parser.add_argument('--model_path',default="",help='load model path')
    parser.add_argument('--mega_noise',default='/data/notebook/work/face/datasets/megaface/megaface/data/megaface_noises.txt',help='megaface_noises.txt path')
    parser.add_argument('--scrub_noise',default='/data/notebook/work/face/datasets/megaface/megaface/data/facescrub_noises.txt',help='facescrub_noises.txt path')
    parser.add_argument('--mega_lst',default='/data/notebook/work/face/datasets/megaface/megaface/data/megaface_lst')
    parser.add_argument('--scrub_lst',default='/data/notebook/work/face/datasets/megaface/megaface/data/facescrub_lst')
    parser.add_argument('--save_base',default='/data/notebook/work/megaface_experiments/save_features')

    args = parser.parse_args()
    return args

def gen_features_run(model,network,scrub_dir,scrub_lst,mega_dir,mega_lst,save_base,img_cuda,batch_size=128):
    # facescrub
    gen_feature_model('facescrub',model,network,scrub_dir,scrub_lst,save_base,img_cuda,batch_size)
    # megaface
    gen_feature_model('megaface',model,network,mega_dir,mega_lst,save_base,img_cuda,batch_size)



if __name__ == '__main__':
    args = parse_args()

    args.save_base = f'{args.save_base}_{args.network}'

    if not os.path.exists(args.save_base):
        os.makedirs(args.save_base)

    gen_feature('facescrub',args.scrub_dir,args.scrub_lst,args.network,args.model_path,args.save_base)
    gen_feature('megaface',args.mega_dir,args.mega_lst,args.network,args.model_path,args.save_base)
