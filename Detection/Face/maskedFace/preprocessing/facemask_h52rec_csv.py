import os
import numpy as np
import cv2
import h5py

import mxnet as mx
from recbuilder import RecBuilder

from tqdm import tqdm
import pandas as pd

path_format = "mask_classification/FaceMask/all/facemask_{}.h5"
csv_format = "FaceMask/images/{}.csv"
save_path = "mask_classification/FaceMask/filtering/"
save_list="mask_classification/FaceMask/filtering/saved_{}.csv"

for t in ['train','val']:
    path = path_format.format(t)
    hf = h5py.File(path,'r')
    images = hf['image']
    targets = hf['target']
    ids = hf['id']

    csv_path = csv_format.format(t)
    df = pd.read_csv(csv_path)

    save_list_path = save_list.format(t)

    wrec = RecBuilder(save_path,tv=t)

    with open(save_list_path,'w') as f:
        f.writelines("ori_id,ori_path,gender,gs,mask,new_id,new_name,mask_type\n")
        for index,val in enumerate(tqdm(df.values)):
            
            ori_id,ori_path,gender,gs,mask,new_id,new_name,mask_type = val

            if gs=='REO': continue
            if gs=='STD' and 'STD3' in ori_path: continue
            if int(mask_type)==2: continue

            save_line = "{},{},{},{},{},{},{},{}\n".format(ori_id,ori_path,gender,gs,mask,new_id,new_name,mask_type)
            f.writelines(save_line)

            image_rgb = images[index]
            image_bgr = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR)
            lb = int(targets[index])
            lb_id = int(ids[index])

            if not int(new_id)==lb_id:
                print("differ...")
                print(index,val)
                exit()
            new_lb = [lb,lb_id]
            wrec.add_image(image_bgr,new_lb)
