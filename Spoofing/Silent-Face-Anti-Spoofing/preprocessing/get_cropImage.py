import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor # For multi processing

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



def process(line,scale=1.0,out_w=80,out_h=80,prefix='/data/'):
    sp = line.strip().split(",")
    ori_path,new_path,ori_w,ori_h,idnum,age,gender,mask,bbox_str,label = sp

    img = cv2.imread(prefix+ori_path)

    bbox_ori = [int(v) for v in bbox_str.split("_")]

    xmin = bbox_ori[0]
    ymin = bbox_ori[1]
    xmax = bbox_ori[2]
    ymax = bbox_ori[3]
    box_w = (xmax-xmin)
    box_h = (ymax-ymin)
    bbox = [xmin,ymin,box_w,box_h]

    new_bbox = crop.crop(img, bbox, scale, out_w, out_h, crop=True)
    make_size = "{}_{}x{}".format(scale,out_w,out_h)

    save_path = "/"+os.path.join(os.path.join(*new_path.split("/")[:-3]),make_size,os.path.join(*new_path.split("/")[-2:]))
    cv2.imwrite(save_path,new_bbox)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out_w", type=int, default=80)
    parser.add_argument("--out_h", type=int, default=80)
    args = parser.parse_args()

    return args


args = get_args()
crop = CropImage()
scale = args.scale
out_w = args.out_w
out_h = args.out_h

print("Scale:",scale," out_w:",out_w," out_h:",out_h)
make_size = "{}_{}x{}".format(scale,out_w,out_h)

tv = ['Training','Validation']
for t in tv:
    path = "/{}/image_list_org_1_80x60.txt".format(t,make_size)
    lines = open(path,'r').readlines()

    make_base = "/{}/{}/{}/"
    for c in ['0','1','2','3']:
        os.makedirs(make_base.format(t,make_size,c))

    with ProcessPoolExecutor(12) as exe:
        _ = [exe.submit(process,line,scale,out_w,out_h) for line in tqdm(lines)]