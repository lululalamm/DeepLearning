import os
import numpy as np
import cv2
import time

from .common import Human
from ..utils.utils_attribute.gender_util import crop_box

# from human.common import Human
# from utils.utils_attribute.gender_util import crop_box


def get_detection(model,img,person=False):
    bboxes, scores, labels = model.detect(img,person=person)

    human = Human(bboxes=bboxes,scores=scores,labels=labels)

    return human

def get_tracking(model,img,human,return_type=3,need_init=False):
    if need_init:
        model.init_tracker()

    result_all = model.track(img,human['bboxes'],human['scores'],return_type=return_type)
    if return_type==1:
        human.tracking_image = result_all
    elif return_type==2:
        online_tlwhs, online_ids, online_scores = result_all
        human.tracking_bboxes = np.array(online_tlwhs)
        human.tracking_ids = online_ids
        human.tracking_scores = online_scores
    else: # 3
        online_tlwhs, online_ids, online_scores, dimg = result_all
        human.tracking_image = dimg
        human.tracking_bboxes = np.array(online_tlwhs)
        human.tracking_ids = online_ids
        human.tracking_scores = online_scores
    return human


def get_gender(model,img,human):
    genders=[]
    genders_str=[]
    for i in range(len(human['bboxes'])):
        bbox = human['bboxes'][i]
        score = human['scores'][i]
        label = human['labels'][i]

        cimg = img.copy()
        crop_person = crop_box(cimg,bbox)
        out = model.infer_image(crop_person)[0]
        pred = np.argmax(out)
        if pred==0:
            pred_str = 'female'
        else:
            pred_str = 'male'

        genders.append(pred)
        genders_str.append(pred_str)

    human.genders = genders
    human.genders_str = genders_str

    return human

def get_pose(model,img,human,box_format='xyxy',use_tracking=False):
    if use_tracking:
        results = model.infer(img,human['tracking_bboxes'],human['tracking_scores'],box_format='xywh')
    else:
        results = model.infer(img,human['bboxes'],human['scores'],box_format=box_format)

    result_kps=[]
    result_kpscore=[]

    for ri in range(len(results[0])):
        result_kps.append(results[0][ri])
        result_kpscore.append(results[1][ri])
    result_kps=np.array(result_kps)
    result_kpscore=np.array(result_kpscore)

    human.pose = result_kps
    human.pose_score = result_kpscore

    return human




