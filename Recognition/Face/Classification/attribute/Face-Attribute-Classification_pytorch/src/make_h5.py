import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tqdm import tqdm

csv_format = "celebA/csv/230720/celebA_anno_mixed10000_{}.csv"

save_format = "celeba_crop_224_{}.h5"
fail_txt = "celeba_crop_224_fail.txt"

align_base = "img_crop224_celeba_scrfd/"

fail_list =[]
with open(fail_txt,'w') as f:
    for tv in ['train','val','test']:
        csv_path = csv_format.format(tv)
        df = pd.read_csv(csv_path)
        
        save_path = save_format.format(tv)
        
        label_list = list(df.keys())
        del label_list[1]
        del label_list[1]
        del label_list[1]
        
        label_list = [label_list[-1],*label_list[1:-1]]
        
        save_images=[]
        save_labels=[]

        for val in tqdm(df.values):
            img_path = os.path.join(align_base,val[0])
            if not os.path.exists(img_path):
                fail_list.append(img_path)
                f.writelines(img_path+"\n")
                continue
            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

            save_images.append(img)
            save_labels.append([val[-1],*val[4:-1]])
            '''
                ['Beard',
                'Smiling',
                'Eyeglasses',
                'Wearing_Lipstick',
                'Wearing_Hat',
                'Wearing_Earrings',
                'Wearing_Necklace',
                'Wearing_Necktie'
                ]
            '''
        
        dt = h5py.special_dtype(vlen=str)

        hf = h5py.File(save_path,'w')
        hf.create_dataset('images', data=np.array(save_images,dtype=np.uint8))
        hf.create_dataset('labels',data=np.array(save_labels,dtype=np.uint8))
        hf.create_dataset('label_list',data=np.array(label_list,dtype=dt))
        hf.close()

print("fail list")
print(fail_list)