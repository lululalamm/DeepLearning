import os
import numpy as np
import cv2
import pandas as pd

import mxnet as mx
from recbuilder import RecBuilder

from tqdm import tqdm


image_base = "FaceMask/id_balanced/images/"
csv_format = "FaceMask/id_balanced/saved_{}_id_balanced.csv"
save_format = "FaceMask/id_balanced/rec/"

for t in ['train','val']:
    csv_path = csv_format.format(t)
    df = pd.read_csv(csv_path)

    save_path = save_format.format(t)
    wrec = RecBuilder(save_path,tv=t)

    for index,val in enumerate(tqdm(df.values)):
        ori_id,ori_path,gender,gs,mask,new_id,new_name,mask_type = val

        if gs=='REO': continue
        if gs=='STD' and 'STD3' in ori_path: continue
        if int(mask_type)==2: continue

        image_path = os.path.join(image_base,new_name)
        image_bgr = cv2.imread(image_path)

        lb = 1 if mask else 0 # mask 0,1
        lb_id = int(new_id) # id
        new_lb = [lb,lb_id]
        wrec.add_image(image_bgr,new_lb)
