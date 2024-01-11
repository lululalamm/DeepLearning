import os
import numpy as np
import skimage.transform

import os
import os.path as osp
import cv2

from ...utils.util_detection import softmax, distance2bbox, distance2kps



class SCRFD_CV:
    def __init__(self, model_file=None, net=None):
        import cv2
        self.model_file = model_file
        self.net = net
        self.taskname = 'detection'
        if self.net is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.net = cv2.dnn.readNetFromONNX(self.model_file)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()
        
    def _init_vars(self):
        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(output_names)==6:
            self.fmc = 2
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(output_names)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(output_names)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(output_names)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True
            
    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        net_outs = self.net.forward(self.output_names)
#       print(self.output_names)
#         print(len(net_outs))
#         for i in range(len(net_outs)):
#             print(len(net_outs[i]))
#         print(net_outs)
        
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx*fmc]
            bbox_preds = net_outs[idx*fmc+1]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx*fmc+2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i
                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)
                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers
            pos_inds = np.where(scores>=thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list
    
    def detect(self, img, thresh=0.5, input_size = (640,640), max_num=0, metric='default'):
        det_scale = 1.0
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8 )
        pos_y=0
        pos_x=0
        if img.shape[0] < input_size[0] and img.shape[1] < input_size[1]:
            pos_y=(input_size[0]-img.shape[0])//2
            pos_x=(input_size[1]-img.shape[1])//2
            det_img[pos_y:pos_y+img.shape[0], pos_x:pos_x+img.shape[1], :] = img
        elif img.shape[0]==img.shape[1] and img.shape[0]>input_size[0]:
            resize = input_size[0]//4*3
            det_scale = float(resize) / img.shape[0]
            img = cv2.resize(img, (resize,resize))
            pos_y=(input_size[0]-img.shape[0])//2
            pos_x=(input_size[1]-img.shape[1])//2
            det_img[pos_y:pos_y+img.shape[0], pos_x:pos_x+img.shape[1], :] = img
        else:
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio>model_ratio:
                new_height = input_size[1]
                pos_y = 0
                new_width = int(new_height / im_ratio)
                pos_x = (input_size[0]-new_width)//2
            else:
                new_width = input_size[0]
                pos_x = 0
                new_height = int(new_width * im_ratio)
                pos_y = (input_size[1]-new_height)//2
            det_scale = float(new_height) / img.shape[0]
            resized_img = cv2.resize(img, (new_width, new_height))
            det_img[pos_y:pos_y+new_height, pos_x:pos_x+new_width, :] = resized_img
                
        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        if self.use_kps:
            kpss = np.vstack(kpss_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss, det_img, [pos_x, pos_y], det_scale
    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


