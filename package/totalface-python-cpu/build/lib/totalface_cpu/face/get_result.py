import os
import numpy as np
import cv2
import time

from ..utils.util_warp import get_new_bbox_pad, get_new_kps_pad, get_new_bbox_resize, get_new_kps_resize
from ..utils.util_warp import get_new_kps_resize_insight,get_new_kps_pad_retina
from ..utils.util_warp import get_new_bbox_blaze, get_new_point_blaze,get_new_bbox_resize_blaze,get_new_point_resize_blaze
from .common import Face
from ..data.image import read_image,read_torchImage,read_video

def get_detection(model_name,model,img,thresh,**kwargs):

    if not isinstance(thresh,(int,float)):
        return 'thresh must numberic value (int,float): {}'.format(type(thresh))

    time_flag = kwargs.get("time_flag",True)
    input_size=kwargs.get("input_size",(640,640)) # h,w
    time_return = kwargs.get("time_return",False)

    target_size = kwargs.get("target_size",0)
    max_size= kwargs.get("max_size",0)

    ret=[]
    det_sizes=[]
    height_min = kwargs.get("height_min",0)

    # if image depth > 3 ( png )
    if img.shape[-1]>3:
        img=img[...,:3]

    if model_name=='scrfd':
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        st = time.time()
        if target_size==0:
            det, kpss, det_img, [pos_x, pos_y], det_scale, time_dict = model.detect(img,thresh=thresh,input_size=input_size,target_size=target_size,max_size=max_size)
        else:
            det, kpss, det_img, det_scale, time_dict = model.detect(img,thresh=thresh,input_size=input_size,target_size=target_size,max_size=max_size)
            pos_x=0
            pos_y=0
        detect_time = time.time()-st

        post2_start = time.time()
        if det.shape[0]==0:
            if time_return:
                return [],time_dict
            else:
                return []

        for i in range(det.shape[0]):
            score = det[i][4]
            bbox = det[i][:4]

            if thresh>0 and score<thresh:
                det_sizes.append(0)
                continue
            kps = None
            if kpss is not None:
                kps = kpss[i]

            bbox = get_new_bbox_pad(bbox,pos_x,pos_y,det_scale)
            if kps is not None:
                kps = get_new_kps_pad(kps,pos_x,pos_y,det_scale)

            det_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            det_sizes.append(det_size)
            face = Face(bbox=bbox,land5=kps,score=score,detect_time=detect_time)
            if height_min>0 and (bbox[3]-bbox[1])<height_min:
                continue
            ret.append(face)
        post2_end = time.time()

    elif model_name=='blaze':

        input_size=kwargs.get("input_size",None)
        resize_method = kwargs.get("resize_way",'pad') # resize or pad

        st = time.time()
        if resize_method=='resize':
            bboxs, keypoints,scores, scale_x, scale_y,input_size,time_dict = model.detect(img,thresh=thresh,input_size=input_size,resize_method='resize')
        else:
            bboxs, keypoints,scores, det_img, [pos_x, pos_y], det_scale,input_size,time_dict = model.detect(img,thresh=thresh,input_size=input_size,resize_method='pad')
        detect_time = time.time()-st


        post2_start = time.time()
        if bboxs.shape[0]==0:
            if time_return:
                return [],time_dict
            else:
                return []

        for i in range(bboxs.shape[0]):
            score = scores[i]
            bbox = bboxs[i]

            if thresh>0 and score<thresh:
                det_sizes.append(0)
                continue

            kps_ = keypoints[i]
            if resize_method=='resize':
                bbox = get_new_bbox_resize_blaze(bbox,scale_x, scale_y,input_size)
            else:
                bbox = get_new_bbox_blaze(bbox,pos_x,pos_y,det_scale,input_size)
            kps=[]
            for kp in kps_:
                if resize_method=='resize':
                    kps.append(get_new_point_resize_blaze(kp[0],kp[1],scale_x, scale_y,input_size))
                else:
                    kps.append(get_new_point_blaze(kp[0],kp[1],pos_x,pos_y,det_scale,input_size))
            kps = np.array(kps)


            det_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            det_sizes.append(det_size)
            face = Face(bbox=bbox,land=kps,score=score,detect_time=detect_time)
            if height_min>0 and (bbox[3]-bbox[1])<height_min:
                continue
            ret.append(face)

        post2_end = time.time()

    elif model_name=='blaze640':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        target_size = kwargs.get("target_size",0)
        max_size= kwargs.get("max_size",0)

        resize_method = kwargs.get("resize_way",'pad') # resize or pad

        st = time.time()

        if resize_method=='resize':
            det, kpss, scale_x, scale_y, time_dict = model.detect(img,thresh=thresh,input_size=input_size,target_size=target_size,max_size=max_size,resize_method='resize')
        else:
            det, kpss,det_img, [pos_x, pos_y], det_scale, time_dict = model.detect(img,thresh=thresh,input_size=input_size,target_size=target_size,max_size=max_size,resize_method='pad')

        detect_time = time.time()-st

        if det.shape[0]==0:
            if time_return:
                return [],time_dict
            else:
                return []

        post2_start = time.time()
        for i in range(det.shape[0]):
            score = det[i][4]
            bbox = det[i][:4]

            if thresh>0 and score<thresh:
                det_sizes.append(0)
                continue
            kps = None
            if kpss is not None:
                kps = kpss[i]

            if resize_method=='resize':
                bbox = get_new_bbox_resize(bbox,scale_x, scale_y)
            else:
                bbox = get_new_bbox_pad(bbox,pos_x,pos_y,det_scale)

            if kps is not None:
                if resize_method=='resize':
                    kps = get_new_kps_resize(kps,scale_x, scale_y)
                else:
                    kps = get_new_kps_pad_retina(kps,pos_x,pos_y,det_scale)


            det_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            det_sizes.append(det_size)
            face = Face(bbox=bbox,land5=kps,score=score,detect_time=detect_time)
            if height_min>0 and (bbox[3]-bbox[1])<height_min:
                continue
            ret.append(face)
        post2_end = time.time()
    

    elif model_name=='dlib':
        st = time.time()
        det, scores, kpss = model.detect(img,thresh=thresh,input_size=input_size)
        detect_time = time.time()-st

        if det.shape[0]==0:
            return []

        for i in range(det.shape[0]):
            score = scores[i]
            bbox = det[i]

            if thresh>0 and score<thresh:
                det_sizes.append(0)
                continue
            kps = None
            if kpss is not None:
                kps = kpss[i]

            det_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            det_sizes.append(det_size)
            face = Face(bbox=bbox,land5=kps,score=score,detect_time=detect_time)
            if height_min>0 and (bbox[3]-bbox[1])<height_min:
                continue
            ret.append(face)

    elif model_name in ['retinaface_torch','retinaface_insightface']:
        # retinaface input is BGR
        if model_name=='retinaface_torch':
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        target_size = kwargs.get("target_size",0)
        max_size= kwargs.get("max_size",0)

        resize_method = kwargs.get("resize_way",'resize') # resize or pad

        st = time.time()

        if resize_method=='resize':
            det, kpss, scale_x, scale_y, time_dict = model.detect(img,thresh=thresh,input_size=input_size,target_size=target_size,max_size=max_size,resize_method='resize')
        else:
            det, kpss,det_img, [pos_x, pos_y], det_scale, time_dict = model.detect(img,thresh=thresh,input_size=input_size,target_size=target_size,max_size=max_size,resize_method='pad')

        detect_time = time.time()-st

        if det.shape[0]==0:
            return [],time_dict

        post2_start = time.time()
        for i in range(det.shape[0]):
            score = det[i][4]
            bbox = det[i][:4]

            if thresh>0 and score<thresh:
                det_sizes.append(0)
                continue
            kps = None
            if kpss is not None:
                kps = kpss[i]

            if resize_method=='resize':
                bbox = get_new_bbox_resize(bbox,scale_x, scale_y)
            else:
                bbox = get_new_bbox_pad(bbox,pos_x,pos_y,det_scale)

            if kps is not None:
                if "insightface" in model_name:
                    if resize_method=='resize':
                        kps = get_new_kps_resize_insight(kps,scale_x,scale_y)
                    else:
                        kps = get_new_kps_pad(kps,pos_x,pos_y,det_scale)
                else:
                    if resize_method=='resize':
                        kps = get_new_kps_resize(kps,scale_x, scale_y)
                    else:
                        kps = get_new_kps_pad_retina(kps,pos_x,pos_y,det_scale)


            det_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            det_sizes.append(det_size)
            face = Face(bbox=bbox,land5=kps,score=score,detect_time=detect_time)
            if height_min>0 and (bbox[3]-bbox[1])<height_min:
                continue
            ret.append(face)
        post2_end = time.time()
        
    # det max 
    max_start= time.time()
    max_idx = np.argmax(det_sizes)
    for ri,r in enumerate(ret):
        if ri==max_idx:
            r.max_flag=True
        else:
            r.max_flag=False
    max_end = time.time()

    if time_return:
        get_detect_time = detect_time*1000
        post2_time = (post2_end-post2_start)*1000
        max_det_time = (max_end-max_start)*1000
        #print("get result detect: {} ms".format(get_detect_time))
        #print("post2: {} ms".format(post2_time))
        #print("max: {} ms".format(max_det_time))
        time_dict['get_detect']=get_detect_time
        time_dict['post2']=post2_time
        time_dict['max_det']=max_det_time
        return ret, time_dict

    else:
        return ret


'''
def get_detection(model_name,model,img,thresh,**kwargs):

    time_flag = kwargs.get("time_flag",True)
    input_size=kwargs.get("input_size",(640,640)) # h,w

    if model_name=='scrfd':
        st = time.time()
        det, kpss, det_img, [pos_x, pos_y], det_scale = model.detect(img,thresh=thresh,input_size=input_size)
        detect_time = time.time()-st
    elif model_name=='dlib':
        st = time.time()
        det, scores, kpss = model.detect(img,thresh=thresh,input_size=input_size)
        detect_time = time.time()-st
    elif model_name=='retinaface':
        st = time.time()
        det, kpss, scale_x, scale_y = model.detect(img,thresh=thresh)
        detect_time = time.time()-st
        
    if det.shape[0]==0:
            return []
    
    ret=[]

    det_sizes=[]

    height_min = kwargs.get("height_min",0)

    for i in range(det.shape[0]):
        if model_name in ['scrfd','retinaface']:
            score = det[i][4]
            bbox = det[i][:4]
        elif model_name=='dlib':
            score = scores[i]
            bbox = det[i]
        if thresh>0 and score<thresh:
            det_sizes.append(0)
            continue
        kps = None
        if kpss is not None:
            kps = kpss[i]
            
        if model_name =='scrfd':
            bbox = get_new_bbox(bbox,pos_x,pos_y,det_scale)
            if kps is not None:
                kps = get_new_kps(kps,pos_x,pos_y,det_scale)

        elif model_name =='retinaface':
            bbox = get_new_bbox_retina(bbox,scale_x, scale_y)
            if kps is not None:
                kps = get_new_kps_retina(kps,scale_x, scale_y)

        det_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        det_sizes.append(det_size)
        face = Face(bbox=bbox,land5=kps,score=score,detect_time=detect_time)
        if height_min>0 and (bbox[3]-bbox[1])<height_min:
            continue
        ret.append(face)

    # det max 
    max_idx = np.argmax(det_sizes)
    for ri,r in enumerate(ret):
        if ri==max_idx:
            r.max_flag=True
        else:
            r.max_flag=False
    return ret
'''


def get_landmark(model_name,model,img,faces,**kwargs):
    
    eye_dist = kwargs.get('eye_dist',False)
    if model_name=='3ddfa':
        #param_lst, roi_box_lst = model.get_results(img,faces)
        param_lst=[]
        roi_box_lst=[]
        for face in faces:
            param,roi_box,_ = model.get(img,face,eye_dist=eye_dist)
            param_lst.append(param)
            roi_box_lst.append(roi_box)
            
        return faces, param_lst, roi_box_lst

    elif model_name=='2d106det':
        land_all=[]
        for face in faces:
            pred, _ = model.get(img,face)
            land_all.append(pred)

        return faces, land_all

    elif model_name in ['PIPNet','pipnet','pip','PIP']:
        for face in faces:
            input_size = kwargs.get("input_size",256)
            scale_mul = kwargs.get("scale_mul",1.1)
            _ = model.get(img,face,input_size=input_size,scale_mul=scale_mul)

        return faces

def get_ageGender(model_name,model,img,faces,to_bgr=False,mask_off=False,eye_min=0):
        
    if model_name=='arcface_cmt':
        for face in faces:
            model.get(img,face,to_bgr,mask_off=mask_off,eye_min=eye_min)
            
    return faces

def get_Race(model_name,model,img,faces,to_bgr=False,mask_off=False,eye_min=0):
        
    if model_name=='arcface_race':
        for face in faces:
            model.get(img,face,to_bgr,mask_off=mask_off,eye_min=eye_min)
            
    return faces

def get_features(model_name, model, img, faces,to_bgr=False):

    if model_name=='arcface':
        for face in faces:
            model.get(img,face,to_bgr)

    return faces

def get_features_mask(model_name, model, model_mask, img, faces,to_bgr=False,mask_flag=False):

    if model_name=='arcface':
        for face in faces:
            if np.argmax(face.mask_sf)==1: # mask
                model_mask.get(img,face,to_bgr,mask_flag=mask_flag)
            else:
                model.get(img,face,to_bgr)

    return faces

def get_features_all(model_name, model, model_mask, img, img_mask, faces,to_bgr=False):

    if model_name=='arcface':
        for face in faces:
            model_mask.get(img_mask,face,to_bgr,mask_flag=True)
            model.get(img,face,to_bgr)

    return faces

def get_mask(model,faces,**kwargs):

    for face in faces:
        model.mask_check(face) # face.mask_sf

    return faces

def get_liveness(model,faces,img,mask_off=False,eye_min=0,to_BGR=True,**kwargs):
    for face in faces:
        model.liveness_check(img,face,mask_off=mask_off,eye_min=eye_min,to_BGR=to_BGR)

    return faces

def get_expression(model,faces,mask_off=False,eye_min=0,**kwargs):

    for face in faces:
        model.get(face) # face.expression (expression result string)

    return faces