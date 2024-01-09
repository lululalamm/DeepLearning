import os
import numpy as np
import cv2

from ...data.image import read_image
# from data.image import read_image

def normalization(rgb_img,mean_list=[0.485, 0.456, 0.406],std_list=[0.229, 0.224, 0.225]):
    MEAN = 255 * np.array(mean_list)
    STD = 255 * np.array(std_list)
    rgb_img = rgb_img.transpose(-1, 0, 1)
    norm_img = (rgb_img - MEAN[:, None, None]) / STD[:, None, None]
    
    return norm_img

def preprocessing(img,img_size,ori_return=False): # RGB input, img_size (h,w)
    if type(img)==str:
        img = read_image(img)
    if ori_return:
        ori_img = img.copy()
    img = cv2.resize(img,(img_size[1],img_size[0]))
    img = normalization(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img,axis=0)
    if ori_return:
        return img,ori_img
    else:
        return img

def crop_box(img,bbox):
    bbox = np.array(bbox,dtype=np.int32)
    crop_person = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

    return crop_person


# ori img is not crop person image (or image path)
# ori img -> detection bboxes -> crop -> infer gender model -> draw
def draw_detect_infer(ori_img,dt_model,gender_model,female_color=(255,0,0),male_color=(0,0,255)):
    
    # detection
    bboxes,scores,labels = dt_model.detect(ori_img)

    if type(ori_img)==str:
        ori_img = read_image(ori_img)
    dimg = ori_img.copy()
    for bbox in bboxes:
        bbox = np.array(bbox,dtype=np.int32)

        cimg = ori_img.copy()
        crop_person = crop_box(cimg,bbox)

        out = gender_model.infer_image(crop_person)[0]
        pred = np.argmax(out)

        if pred==0:
            color = female_color
        else:
            color = male_color

        dimg = cv2.rectangle(dimg,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)

    return dimg

# ori img is not crop person image (or image path)
# bboxes -> crop -> infer gender model -> draw
def draw_detect_bboxes(ori_img,bboxes,gender_model,female_color=(255,0,0),male_color=(0,0,255)):

    if type(ori_img)==str:
        ori_img = read_image(ori_img)
    dimg = ori_img.copy()
    for bbox in bboxes:
        bbox = np.array(bbox,dtype=np.int32)
        
        cimg = ori_img.copy()
        crop_person = crop_box(cimg,bbox)
        
        out = gender_model.infer_image(crop_person)[0]
        pred = np.argmax(out)

        if pred==0:
            color = female_color
        else:
            color = male_color

        dimg = cv2.rectangle(dimg,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)

    return dimg

def draw_results(ori_img,bboxes,genders,female_color=(255,0,0),male_color=(0,0,255)):
    for i,bbox in enumerate(bboxes):
        gender = genders[i]
        bbox = np.array(bbox,dtype=np.int32)

        if gender==0:
            color = female_color
        else:
            color = male_color

        dimg = cv2.rectangle(ori_img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)

    return dimg

def draw_results_human(ori_img,human,female_color=(255,0,0),male_color=(0,0,255)):
    bboxes = human['bboxes']
    genders = human['genders']
    
    for i,bbox in enumerate(bboxes):
        gender = genders[i]
        bbox = np.array(bbox,dtype=np.int32)

        if gender==0:
            color = female_color
        else:
            color = male_color

        dimg = cv2.rectangle(ori_img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)

    return dimg

        