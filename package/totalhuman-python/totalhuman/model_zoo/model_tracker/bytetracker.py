import os
import numpy as np
from loguru import logger
import easydict
import cv2

from .bytetrack.visualize import plot_tracking, vis, _COLORS
from .bytetrack.tracker.byte_tracker import BYTETracker
from .bytetrack.tracking_utils.timer import Timer

from ...human.get_result import get_detection,get_gender
# from human.get_result import get_detection,get_gender

class ByteTracker:
    def __init__(self,**kwargs):
        self.frame_rate = kwargs.get('frame_rate',30)
        self.init_args(**kwargs)
        self.init_tracker()

    def init_args(self,**kwargs):
        self.aspect_ratio_thresh = kwargs.get('aspect_ratio_thresh',1.6)
        self.min_box_area = kwargs.get('min_box_area',10)
        self.track_thresh = kwargs.get('track_thresh',0.5)
        self.track_buffer = kwargs.get('track_buffer',60)
        self.match_thresh = kwargs.get('match_thresh',0.8)
        self.mot20 = kwargs.get('mot20',False)

        self.args = easydict.EasyDict({
                'aspect_ratio_thresh' : self.aspect_ratio_thresh, 
                'min_box_area' : self.min_box_area,
                'track_thresh' : self.track_thresh, # tracking confidence threshold
                'track_buffer' : self.track_buffer, # the frames for keep lost tracks
                'match_thresh' : self.match_thresh, # matching threshold for tracking
                'mot20' : self.mot20})

    def init_tracker(self,**kwargs):
        self.init_args(**kwargs)
        self.frame_rate = kwargs.get('frame_rate',30)

        self.tracker = BYTETracker(self.args,frame_rate=self.frame_rate)
        self.frame_id = 0
        self.timer = Timer()
        print("tracker init")

    # image - original (raw) image
    # return type : 1 = draw image / 2 = results / 3 = all
    def dttrack(self,dt_model,image,toRGB=False,person=False,return_type=1,**kwargs): 

        draw_info = kwargs.get('draw_info',True)
        draw_box = kwargs.get('draw_box',True)
        draw_id = kwargs.get('draw_id',True)
        draw_blur = kwargs.get('draw_blur',False)
        min_size = kwargs.get('min_size',0)

        # detect
        self.timer.tic()
        if toRGB:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        bboxes,scores,labels = dt_model.detect(image,person=person) # rgb image array or path

        if labels is not None:
            online_tlwhs = []
            online_ids = []
            online_scores = []
            #results = []

            
            online_targets = self.tracker.update(bboxes,scores)

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    online_tlwhs.append(tlwh) # box
                    online_ids.append(tid) # id number
                    online_scores.append(t.score) # confidence
                    # results.append(
                    #     f"{self.frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    # )   
        
            self.timer.toc()

            if return_type==1:
                online_im = plot_tracking(image, online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=1. / self.timer.average_time, \
                                            draw_info=draw_info,draw_box=draw_box,draw_id=draw_id,draw_blur=draw_blur,min_size=min_size) 
                self.frame_id+=1
                return online_im
            elif return_type==2:
                self.frame_id+=1
                return online_tlwhs, online_ids, online_scores
            else:
                online_im = plot_tracking(image, online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=1. / self.timer.average_time, \
                                            draw_info=draw_info,draw_box=draw_box,draw_id=draw_id,draw_blur=draw_blur,min_size=min_size) 
                self.frame_id+=1
                return online_tlwhs, online_ids, online_scores, online_im
            
        else:
            self.timer.toc()
            if return_type==1:
                return image
            elif return_type==2:
                return [],[],[],[]
            else:
                return [],[],[],[],image

    # return type : 1 = draw image / 2 = results / 3 = all
    def track(self,image,bboxes,scores,return_type=1,**kwargs): # image - original (raw) image
        online_tlwhs = []
        online_ids = []
        online_scores = []
        #results = []

        draw_info = kwargs.get('draw_info',True)
        draw_box = kwargs.get('draw_box',True)
        draw_id = kwargs.get('draw_id',True)
        draw_blur = kwargs.get('draw_blur',False)
        min_size = kwargs.get('min_size',0)

        self.timer.tic()
        online_targets = self.tracker.update(bboxes,scores)

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(tlwh) # box
                online_ids.append(tid) # id number
                online_scores.append(t.score) # confidence
                # results.append(
                #     f"{self.frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                # )   
        
        self.timer.toc()

        if return_type==1:
            online_im = plot_tracking(image, online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=1. / self.timer.average_time, \
                                        draw_info=draw_info,draw_box=draw_box,draw_id=draw_id,draw_blur=draw_blur,min_size=min_size)
            self.frame_id+=1
            return online_im
        elif return_type==2:
            self.frame_id+=1
            return online_tlwhs, online_ids, online_scores
        else:
            online_im = plot_tracking(image, online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=1. / self.timer.average_time, \
                                        draw_info=draw_info,draw_box=draw_box,draw_id=draw_id,draw_blur=draw_blur,min_size=min_size) 
            self.frame_id+=1
            return online_tlwhs, online_ids, online_scores, online_im



        








    

