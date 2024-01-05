import os
import numpy as np
import argparse

import h5py
import cv2
from tqdm import tqdm



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_format", type=str, default="FaceMask/images/{}.txt")
    parser.add_argument("--image_base", type=str, default='FaceMask/images/all/')
    parser.add_argument("--save_format",type=str, default="facemask_{}.h5")
    args = parser.parse_args()

    return args

args = get_args()
txt_format = args.txt_format
image_base = args.image_base
save_format = args.save_format


for t in ['train','val']:
    txt_path = txt_format.format(t)
    lines = open(txt_path,'r').readlines()

    save_path = save_format.format(t)
    save_hf = h5py.File(save_path,'w')
    # save_hf.create_dataset("image", (len(lines),112,112,3), dtype='uint8')
    # save_hf.create_dataset("target", (len(lines),), dtype='uint8')
    # save_hf.create_dataset("id",(len(lines),),dtype='int32')

    images=[]
    targets=[]
    idnames=[]

    for li,line in enumerate(tqdm(lines)):
        image_path = os.path.join(image_base,line.strip())
        img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB).astype(np.uint8)

        mask_flag = line.strip().split(".jpg")[0].split("_")[-1]
        if mask_flag=='unmask':
            target=0
        else:
            target=1
        idname = int(line.strip().split("_")[1])

        images.append(img)
        targets.append(target)
        idnames.append(idname)

        # save_hf["image"][li,:,:,:] = np.array(img).tolist()
        # save_hf["target"][li] = target
        # save_hf['id'][li] = idname

    save_hf.create_dataset("image", data=np.array(images), dtype=np.uint8)
    save_hf.create_dataset("target", data=np.array(targets), dtype=np.uint8)
    save_hf.create_dataset("id",data=np.array(idnames),dtype=np.int32)
    

    save_hf.close()
    print("Save!",save_path)

