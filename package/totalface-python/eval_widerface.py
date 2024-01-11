"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""
# modified

# box format x1 y1 w h score

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat

from .model_zoo.get_models import get_detection_model,get_landmark_model, get_ageGender_model,get_recognition_model
from .face.get_result import *
from .utils.util_common import draw_result,draw_result_sim
from .data.image import read_image

import time

def bbox_overlaps(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] +
                          1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] +
                                1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

def get_preds_pkl(save_pred):
    with open(save_pred,'rb') as f:
        boxes = pickle.load(f)

    return boxes

def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip().split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds_txt(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = np.array(_boxes,dtype=np.float)
        boxes[event] = current_event
    return boxes
        

def get_preds(pred_dir,detection_model,thresh,input_size):
    events = [ name for name in os.listdir(pred_dir) if not ".DS" in name ]
    boxes = dict()
    pbar = tqdm.tqdm(events)
    all_times=0
    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = [ name for name in os.listdir(event_dir) if not ".DS" in name]
        current_event = dict()
        sub_times=0
        for imgname in event_images:
            img_path = os.path.join(event_dir,imgname)
            
            img = read_image(img_path)
            st = time.time()
            faces = get_detection(detection_name,detection_model,img,thresh=thresh,input_size=input_size)
            ed = time.time()
            _boxes=[]

            for face in faces:
                x1 = face.bbox[0]
                y1 = face.bbox[1]
                w = face.bbox[2]-face.bbox[0]
                h = face.bbox[3]-face.bbox[1]
                _boxes.append([x1,y1,w,h,face.score])


            #dt_times = face.detect_time
            dt_times = (ed-st)
            sub_times+=dt_times

            _boxes = list(sorted(np.array(_boxes),key=lambda x:x[4] ,reverse=True)) ## score 기준 내림차순 정렬

            current_event[imgname.split(".")[0]] = np.array(_boxes,dtype=np.float)
            del faces
        sub_times = sub_times/len(event_images)
        all_times +=sub_times

        boxes[event] = current_event
    all_times = all_times/len(events)

    print(">> Detect time mean: {} ms".format(all_times))
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore: 1이면 gt 존재
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0: # 가장 많이 겹치는 박스가 ignore 해야하면
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list
    


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0] 
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index) # thresh 넘고, ignore 가 아니면서 내가 predict 한 index -> tp+fp
            pr_info[t, 1] = pred_recall[r_index] # tp
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0] # tp / (tp+fp)
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face # tp / (tp+fn)

    # tp : pr_curve[i,1]
    # fp : pr_curve[i,0] - pr_curve[i,1]
    # tn : 
    # fn : count_face - pr_curve[i,1]
    return _pr_curve

def get_threshs(thresh_num):
    threshs=[]
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        threshs.append(thresh)
    return np.array(threshs)



def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, detection_model,input_size,save_result,thresh,save_pred,use_txt=True,iou_thresh=0.5):
    if use_txt:
        pred = get_preds_txt(save_pred)
        print("Load pred:",save_pred)
    else:
        if os.path.exists(save_pred):
            pred = get_preds_pkl(save_pred)
            print("Load pred:",save_pred)
        else:
            pred = get_preds(pred,detection_model,thresh,input_size)
            with open(save_pred,'wb') as f:
                pickle.dump(pred,f)
                print("Save pred:",save_pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    save_dict={}
    for setting_id in range(3):
        save_dict[settings[setting_id]]={}
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0: # 해당 level 에서 인정되는 gt box 가 있다면
                    ignore[keep_index-1] = 1 # 인정되는 box index 에 1 을 줌
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

        threshs = get_threshs(thresh_num)

    
        save_dict[settings[setting_id]]['recall']=recall
        save_dict[settings[setting_id]]['precision']=propose
        save_dict[settings[setting_id]]['ap']=ap
        save_dict[settings[setting_id]]['thresh']=threshs

    
    with open(save_result.split(".pkl")[0]+"_WF.txt",'w') as f:
        new_line = "==================== Results ===================="
        f.writelines(new_line+"\n")
        print(new_line)

        new_line = "Easy   Val AP: {}".format(aps[0])
        f.writelines(new_line+"\n")
        print(new_line)

        new_line = "Medium Val AP: {}".format(aps[1])
        f.writelines(new_line+"\n")
        print(new_line)

        new_line = "Hard   Val AP: {}".format(aps[2])
        f.writelines(new_line+"\n")
        print(new_line)

        new_line = "================================================="
        f.writelines(new_line+"\n")
        print(new_line)

    with open(save_result, 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    print("Save {}".format(save_result))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="WiderFace/WIDER_val/images/")
    parser.add_argument('-g', '--gt', default="WiderFace/ground_truth/")
    parser.add_argument('--dt_name',type=str,default='scrfd')
    parser.add_argument('--dt_path',type=str,default='scrfd_10g_bnkps.v8.trt')
    parser.add_argument('--dt_path2',type=str,default='')
    parser.add_argument('--thresh',type=float,default=0.02)
    parser.add_argument('--save_format',type=str,default="result_ap_txt/result_WF_{}.pkl")
    parser.add_argument('--pred_format',type=str,default="result_txt/result_WF_{}/")#default="save_pred/pred_WF_{}.pkl")
    parser.add_argument('--input_size',nargs='+', type=int, default=[640,640],help="input size height, width")
    parser.add_argument('--use_txt',type=lambda x: (str(x).lower() == 'true'),default=True)
    args = parser.parse_args()

    detection_name= args.dt_name
    detection_path = args.dt_path
    detection_path2 = args.dt_path2
    detection_thresh = args.thresh
    detection_size = tuple(args.input_size)

    save_result = args.save_format.format(detection_path.split("/")[-1])
    os.makedirs(os.path.join(*save_result.split("/")[:-1]), exist_ok=True)
    save_pred = args.pred_format.format(detection_path.split("/")[-1])

    if args.use_txt:
        detection_model = None
    else:
        if detection_path2:
            detection_path = [detection_path,detection_path2]
        detection_model = get_detection_model(detection_name,detection_path)

    evaluation(args.pred, args.gt,detection_model,detection_size,save_result,detection_thresh,save_pred,args.use_txt)