import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob

#from utils.utils_detection.yolo_util import xyxy2xywh,labels2coco
from ..utils.utils_detection.yolo_util import xyxy2xywh,labels2coco

categories = [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'car'}, {'id': 4, 'name': 'motorcycle'}, {'id': 5, 'name': 'airplane'}, \
    {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'train'}, {'id': 8, 'name': 'truck'}, {'id': 9, 'name': 'boat'}, {'id': 10, 'name': 'traffic light'}, \
    {'id': 11, 'name': 'fire hydrant'}, {'id': 12, 'name': ''}, {'id': 13, 'name': 'stop sign'}, {'id': 14, 'name': 'parking meter'}, {'id': 15, 'name': 'bench'}, \
    {'id': 16, 'name': 'bird'}, {'id': 17, 'name': 'cat'}, {'id': 18, 'name': 'dog'}, {'id': 19, 'name': 'horse'}, {'id': 20, 'name': 'sheep'}, {'id': 21, 'name': 'cow'}, \
    {'id': 22, 'name': 'elephant'}, {'id': 23, 'name': 'bear'}, {'id': 24, 'name': 'zebra'}, {'id': 25, 'name': 'giraffe'}, {'id': 26, 'name': ''}, {'id': 27, 'name': 'backpack'}, \
    {'id': 28, 'name': 'umbrella'}, {'id': 29, 'name': ''}, {'id': 30, 'name': ''}, {'id': 31, 'name': 'handbag'}, {'id': 32, 'name': 'tie'}, {'id': 33, 'name': 'suitcase'}, \
    {'id': 34, 'name': 'frisbee'}, {'id': 35, 'name': 'skis'}, {'id': 36, 'name': 'snowboard'}, {'id': 37, 'name': 'sports ball'}, {'id': 38, 'name': 'kite'}, \
    {'id': 39, 'name': 'baseball bat'}, {'id': 40, 'name': 'baseball glove'}, {'id': 41, 'name': 'skateboard'}, {'id': 42, 'name': 'surfboard'}, {'id': 43, 'name': 'tennis racket'}, \
    {'id': 44, 'name': 'bottle'}, {'id': 45, 'name': ''}, {'id': 46, 'name': 'wine glass'}, {'id': 47, 'name': 'cup'}, {'id': 48, 'name': 'fork'}, {'id': 49, 'name': 'knife'}, \
    {'id': 50, 'name': 'spoon'}, {'id': 51, 'name': 'bowl'}, {'id': 52, 'name': 'banana'}, {'id': 53, 'name': 'apple'}, {'id': 54, 'name': 'sandwich'}, {'id': 55, 'name': 'orange'}, \
    {'id': 56, 'name': 'broccoli'}, {'id': 57, 'name': 'carrot'}, {'id': 58, 'name': 'hot dog'}, {'id': 59, 'name': 'pizza'}, {'id': 60, 'name': 'donut'}, {'id': 61, 'name': 'cake'}, \
    {'id': 62, 'name': 'chair'}, {'id': 63, 'name': 'couch'}, {'id': 64, 'name': 'potted plant'}, {'id': 65, 'name': 'bed'}, {'id': 66, 'name': ''}, {'id': 67, 'name': 'dining table'}, \
    {'id': 68, 'name': ''}, {'id': 69, 'name': ''}, {'id': 70, 'name': 'toilet'}, {'id': 71, 'name': ''}, {'id': 72, 'name': 'tv'}, {'id': 73, 'name': 'laptop'}, {'id': 74, 'name': 'mouse'}, \
    {'id': 75, 'name': 'remote'}, {'id': 76, 'name': 'keyboard'}, {'id': 77, 'name': 'cell phone'}, {'id': 78, 'name': 'microwave'}, {'id': 79, 'name': 'oven'}, {'id': 80, 'name': 'toaster'}, \
    {'id': 81, 'name': 'sink'}, {'id': 82, 'name': 'refrigerator'}, {'id': 83, 'name': ''}, {'id': 84, 'name': 'book'}, {'id': 85, 'name': 'clock'}, {'id': 86, 'name': 'vase'}, \
    {'id': 87, 'name': 'scissors'}, {'id': 88, 'name': 'teddy bear'}, {'id': 89, 'name': 'hair drier'}, {'id': 90, 'name': 'toothbrush'}]

coco_cls_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush']

def coco_eval(dt_model,data_dir,resFile):
    pathlist = glob.glob(os.path.join(data_dir, '*.jpg'))
    inputs_num = len(pathlist)
    print("total {} images".format(inputs_num))

    image_ids = []
    detection_boxes_list = []
    detection_classes_list = []
    detection_scores_list = []

    print("conf thres change: {} to 0.01 for eval".format(dt_model.conf_thres))
    dt_model.conf_thres=0.01

    i=0
    for input_file in pathlist:
        bboxes,scores,labels = dt_model.detect(input_file) # image path input
        if len(labels)<1:
            bboxes_xywh = bboxes
            new_labels = labels
        else:
            bboxes_xywh = xyxy2xywh(bboxes)
            new_labels = labels2coco(labels,new_cls=coco_cls_list)
        img_id = int(os.path.splitext(os.path.basename(input_file))[0])    
        image_ids.append(img_id)
        detection_classes_list.append(new_labels+1)
        detection_boxes_list.append(bboxes_xywh)    
        detection_scores_list.append(scores)
            
        i+=1
        if i%500==0:
            print("Done {}/{}...".format(i,inputs_num))
        ## for beauty, use tqdm..?

    #from ..data.coco_tools import ExportDetectionsToCOCO
    from data.coco_tools import ExportDetectionsToCOCO
    # Save result json to "output_path"  ex) './output.json'
    ExportDetectionsToCOCO(image_ids, detection_boxes_list, detection_scores_list, detection_classes_list, categories, output_path=resFile)


def get_map(annFile,resFile,person=False):
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print('Running demo for *%s* results.'%(annType))

    #annFile = './instances_val2017.json'
    cocoGt=COCO(annFile)

    #resFile = './output.json'
    cocoDt=cocoGt.loadRes(resFile)
        
    imgIds = sorted(cocoGt.getImgIds())
    #imgIds = imgIds[0:100]
    #imgId  = imgIds[np.random.randint(maxdets)]

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    if person is True:
        cocoEval.params.catIds = [1]
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()