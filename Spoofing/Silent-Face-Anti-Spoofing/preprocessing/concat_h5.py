import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=2.7)
    args = parser.parse_args()

    return args

#slack_api = SlackApi()

args = get_args()
scale = args.scale
if scale==4:
    scale = int(scale)
else:
    scale = float(scale)


concat1 = "FaceInTheWild/unmask/Training/{}_80x80.h5".format(scale)
concat2 = "CelebA_Spoof/Training/{}_80x80.h5".format(scale)
save = "wild_celebA/"
if not os.path.exists(save):
    os.makedirs(save)
save = "wild_celebA/{}_80x80.h5".format(scale)



keys=['image','target']


save_hf = h5py.File(save,'w')
hf1 = h5py.File(concat1,'r')
hf2 = h5py.File(concat2,'r')

count = len(hf1['image'])+len(hf1['image'])

save_hf.create_dataset("image", (count,80,80,3), dtype='uint8')
save_hf.create_dataset("target", (count,), dtype='uint8')
    
images1 = hf1['image']
targets1 = hf1['target']
    
images2 = hf2['image']
targets2 = hf2['target']
print("finish load datas")

c=0
for i in tqdm(range(len(images1))):
    save_hf["image"][c,:,:,:] = np.array(images1[i]).tolist()
    save_hf["target"][c] = targets1[i]
    c+=1

for i in tqdm(range(len(images2))):
    save_hf["image"][c,:,:,:] = np.array(images2[i]).tolist()
    save_hf["target"][c] = targets2[i]
    c+=1

save_hf.close()
