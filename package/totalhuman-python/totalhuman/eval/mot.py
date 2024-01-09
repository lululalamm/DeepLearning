import os
import numpy as np
import cv2
import torch
import easydict

import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


# pkg
from ..model_zoo.model_detection.yolox import YoloX
from ..model_zoo.model_detection.yolov7 import Yolov7
from ..model_zoo.model_detection.yolov5 import Yolov5

from .mot_evaluator_custom import MOTEvaluator
from ..utils.utils_tracker.mot_custom import ValTransform, MOTDataset, get_eval_loader

# code
# from model_zoo.model_detection.yolox import YoloX
# from model_zoo.model_detection.yolov7 import Yolov7
# from model_zoo.model_detection.yolov5 import Yolov5

# from eval.mot_evaluator_custom import MOTEvaluator
# from utils.utils_tracker.mot_custom import ValTransform, MOTDataset, get_eval_loader

# data dir : mix_det/mot/
# json file : mix_det/mot/annotations/train.json
def get_eval(dt_name,dt_model, data_dir, json_file, result_base, name='train',save_name=None, \
            img_size=(640,640), aspect_ratio_thresh=1.6,min_box_area=100,conf=0.001,nmsthre=0.65,batch_size=1,seed=None, \
            track_thresh=0.6, track_buffer=30, match_thresh=0.9, mot20=False, stride=32):
    args = easydict.EasyDict({
    
        'aspect_ratio_thresh' : aspect_ratio_thresh, 
        'min_box_area' : min_box_area, #100
        'conf' : conf,
        'nmsthre' : nmsthre, #0.65
        'batch_size' : batch_size,
        'seed' : seed,
        'track_thresh' : track_thresh, # tracking confidence threshold
        'track_buffer' : track_buffer, # the frames for keep lost tracks
        'match_thresh' : match_thresh, # matching threshold for tracking
        'mot20' : mot20})

    data_dir = data_dir
    json_file = json_file

    img_size=img_size
    result_base = "./mot_eval/"
    if not save_name is None:
        result_folder = os.path.join(result_base,save_name)
    else:
        result_folder = os.path.join(result_base,dt_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # step 1
    val_transform = ValTransform(dt_name=dt_name,stride=stride)
    # step 2
    val_loader = get_eval_loader(batch_size,data_dir,json_file,name=name,testdev=False,transform=val_transform,img_size=img_size)
    # eval 
    mot_eval = MOTEvaluator(args,val_loader)
    eval_result = mot_eval.evaluate(dt_model,result_folder=result_folder)

    return eval_result

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            print('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            print('No ground truth for {}, skipping.'.format(k))

    return accs, names

# gt base : '/data/shared/Human/HumanTracker/datasets/mix_det/mot/train/'
def print_mota(gt_base,result_folder,mot20=False):
    mm.lap.default_solver = 'lap'
    gt_type=''
    if mot20:
        gtfiles = glob.glob(os.path.join(gt_base, '*/gt/gt{}.txt'.format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join(gt_base, '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)

    tsfiles = [f for f in glob.glob(os.path.join(result_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    print('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    print('Available LAP solvers {}'.format(mm.lap.available_solvers))
    print('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    print('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)

    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
           'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
           'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)    

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])

    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                    'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']

    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


