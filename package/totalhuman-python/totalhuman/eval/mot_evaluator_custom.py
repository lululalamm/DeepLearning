from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

# pkg
from ..model_zoo.model_tracker.bytetracker import ByteTracker
from ..utils.utils_tracker.visualize import plot_tracking, vis, _COLORS
from ..utils.utils_tracker.timer import Timer
from ..utils.utils_tracker.dist import gather, is_main_process, synchronize,time_synchronized

# code
# from model_zoo.model_tracker.bytetracker import ByteTracker
# from utils.utils_tracker.visualize import plot_tracking, vis, _COLORS
# from utils.utils_tracker.timer import Timer
# from utils.utils_tracker.dist import gather, is_main_process, synchronize,time_synchronized


#from ..layers import COCOeval_opt as COCOeval
import numpy as np
import sys
sys.path.append('..')

import contextlib
import io
import os
import itertools
import json
import tempfile
import time



def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                score = score.item()
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        summary=True,
        test_save=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate. (detection model, onnx or trt or openvino)

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.FloatTensor#torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        print("detection conf thres {} to {} for mot eval".format(model.conf_thres,self.args.conf))
        print("detection iou thres {} to {} for mot eval".format(model.iou_thres,self.args.nmsthre))
        model.conf_thres = self.args.conf
        model.iou_thres = self.args.nmsthre

        tracker = ByteTracker(aspect_ratio_thresh = self.args.aspect_ratio_thresh ,
                              min_box_area = self.args.min_box_area,
                              track_thresh = self.args.track_thresh,
                              track_buffer = self.args.track_buffer,
                              match_thresh = self.args.match_thresh,
                              mot20 = self.args.mot20)
        ori_thresh = self.args.track_thresh
        if not self.args.batch_size==1:
            print("batch size not 1")
            return

        if test_save:
            test_save_path = os.path.join(result_folder,"test_save.txt")
            test_f = open(test_save_path,'w') 

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = ByteTracker(aspect_ratio_thresh = self.args.aspect_ratio_thresh ,
                              min_box_area = self.args.min_box_area,
                              track_thresh = self.args.track_thresh,
                              track_buffer = self.args.track_buffer,
                              match_thresh = self.args.match_thresh,
                              mot20 = self.args.mot20)
                    try:
                        if len(results) != 0:
                            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                            #result_npy_path = os.path.join(result_folder,'{}.npy'.format(video_names[video_id - 1]))
                            #print("result total:",len(results))
                            #np.save(result_npy_path,results)

                            write_results(result_filename, results)
                            results = []
                    except:
                        print("names:",video_names)
                        print("id:",video_id)
                        print("name:",video_name)
                        exit()
                ###
                input_shape = imgs.numpy().shape[1:]
                raw_shape = (info_imgs[0].item(), info_imgs[1].item())
                img_id = ids[0][0].numpy()
                
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                bboxes,scores,labels = model.detect_moteval(imgs,raw_shape,person=True)

                if test_save:
                    for i in range(len(bboxes)):
                        new_line = "{} {} {} {} {} {} {}\n".format(img_file_name, \
                        bboxes[i][0],bboxes[i][1],bboxes[i][2],bboxes[i][3], \
                        scores[i],labels[i])
                        test_f.writelines(new_line)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            # run tracking
            if labels is not None:
                online_targets = tracker.tracker.update(bboxes,scores)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh) # box
                        online_ids.append(tid) # id number
                        online_scores.append(t.score) # confidence
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            output_results = self.convert_to_coco_format(img_id, bboxes,scores,labels)
            #print(output_results)
            data_list.extend(output_results)

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics,summary=summary)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, img_id, bboxes, scores,labels):
        
        def xyxy2xywh(bboxes):
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
            return bboxes
        
        data_list = []

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            label = self.dataloader.dataset.class_ids[int(labels[ind])]

            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                 "bbox": bboxes[ind].tolist(),
                 "score": scores[ind].item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics,summary=False):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            from pycocotools.cocoeval import COCOeval
            #from ..layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()

            if summary:
                cocoEval.summarize()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
