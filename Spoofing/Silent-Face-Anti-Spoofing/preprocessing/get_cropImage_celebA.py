import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from src.generate_patches import CropImage

from totalface.model_zoo.get_models import get_detection_model
from totalface.face.get_result import get_detection


from tqdm import tqdm
import h5py


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)


            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=2.7)
    parser.add_argument("--out_w", type=int, default=80)
    parser.add_argument("--out_h", type=int, default=80)
    args = parser.parse_args()

    return args

#slack_api = SlackApi()

args = get_args()
crop = CropImage()
scale = args.scale
out_w = args.out_w
out_h = args.out_h
thresh = 0.5
input_size=(640,640)

image_cropper = CropImage()

dt_name='scrfd'
dt_path = "scrfd_10g_bnkps.v8.trt"
dt_model = get_detection_model(dt_name,dt_path,load_multi=False)
detector_confidence = 0.5

print("Scale:",scale," out_w:",out_w," out_h:",out_h)
make_size = "{}_{}x{}".format(scale,out_w,out_h)

image_base = "CelebA_Spoof/"
path = "CelebA_Spoof/metas/intra_test/train_label.txt"
lines = open(path,'r').readlines()

if scale==4:
    scale_name = int(scale)
else:
    scale_name = scale
save_path = "CelebA_Spoof/Training/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = "CelebA_Spoof/Training/{}_{}x{}.h5".format(scale_name,out_w,out_h)
save_hf = h5py.File(save_path,'w')

not_detect = "CelebA_Spoof/Training/not_detect_CelebA.txt"
images=[]
targets=[]

with open(not_detect,'w') as f:
    for li,line in enumerate(tqdm(lines)):
        sp = line.strip().split(" ")
        img_path = os.path.join(image_base,sp[0])
        lb = int(sp[1])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        faces = get_detection(dt_name,dt_model,img_rgb,thresh=thresh,input_size=input_size)
        if len(faces)<1:
            f.writelines(img_path+"\n")
            continue

        bbox_ori = faces[0]['bbox']
        image_bbox = [int(bbox_ori[0]), int(bbox_ori[1]), int(bbox_ori[2]-bbox_ori[0]+1), int(bbox_ori[3]-bbox_ori[1]+1)]

        param = {
        "org_img": img,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": out_w,
        "out_h": out_h,
        "crop": True,
        }

        img = image_cropper.crop(**param).astype(np.uint8)
        images.append(img)
        targets.append(int(lb))

save_hf.create_dataset("image", data=np.array(images), dtype=np.uint8)
save_hf.create_dataset("target", data=np.array(targets), dtype=np.uint8)
save_hf.close()

