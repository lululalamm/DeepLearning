import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from tqdm import tqdm

import multiprocessing



def get_items(val,image_list,label_list,shape_list,fail_list,input_size=224):
    img_path = os.path.join(align_base,val[0])
    if not os.path.exists(img_path):
        fail_list.append(img_path)
        return 
    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    shape_list.append([img.shape])
    img = cv2.resize(img,(input_size,input_size))
    save_images.append(img)
    save_labels.append([val[-1],*val[4:-1]])
    return

def update(*a):
    pbar.update()

csv_format = "celebA/csv/230720/celebA_anno_mixed10000_{}.csv"
save_format = "celeba_crop_224_{}.h5"
fail_txt = "celeba_crop_224_fail_{}.txt"

align_base = "celebA/Img/img_crop224_celeba_scrfd/"

for tv in ['train','val','test']:
    print("Start",tv)
    csv_path = csv_format.format(tv)
    df = pd.read_csv(csv_path)
        
    save_path = save_format.format(tv)
        
    label_list = list(df.keys())
    del label_list[1]
    del label_list[1]
    del label_list[1]
        
    label_list = [label_list[-1],*label_list[1:-1]]
    
    manager    = multiprocessing.Manager()
    save_images=manager.list()
    save_labels=manager.list()
    save_shape=manager.list()
    fail_list =manager.list()

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pbar = tqdm(total=len(df))

    for i in range(pbar.total):
        pool.apply_async(get_items, args=(df.values[i],save_images,save_labels,save_shape,fail_list,224,), callback=update)
        
    pool.close()
    pool.join()
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
    
    save_images = list(save_images)
    save_labels = list(save_labels)
    save_shape = list(save_shape)
    fail_list = list(fail_list)

    dt = h5py.special_dtype(vlen=str)
    hf = h5py.File(save_path,'w')
    hf.create_dataset('images', data=np.array(save_images,dtype=np.uint8))
    hf.create_dataset('labels',data=np.array(save_labels,dtype=np.uint8))
    hf.create_dataset('shapes',data=np.array(save_shape,dtype=np.int32))
    hf.create_dataset('label_list',data=np.array(label_list,dtype=dt))
    hf.close()

    with open(fail_txt.format(tv),'w') as f:
        for fail_path in fail_list:
            f.writelines(fail_path+"\n")

    print("Finish",tv)

