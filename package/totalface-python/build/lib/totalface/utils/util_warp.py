import os
import numpy as np
import skimage.transform
import cv2

def get_new_bbox_blaze(bbox,pos_x, pos_y, det_scale,resize=[256,256]):
    xmin,ymin,xmax,ymax = bbox
    ymin = ((ymin*resize[0])-pos_y)/det_scale
    xmin = ((xmin*resize[1])-pos_x)/det_scale
    ymax = ((ymax*resize[0])-pos_y)/det_scale
    xmax = ((xmax*resize[1])-pos_x)/det_scale
    
    return np.array([xmin,ymin,xmax,ymax]).astype(np.float32)

def get_new_bbox_resize_blaze(det,scale_x,scale_y,resize=[256,256]):

    x1,y1,x2,y2 = det

    det[0] = x1*resize[1]/scale_x
    det[1] = y1*resize[0]/scale_y
    det[2] = x2*resize[1]/scale_x
    det[3] = y2*resize[0]/scale_y

    det = det.astype(np.float32)

    return det

def get_new_point_blaze(px,py,pos_x, pos_y,det_scale,resize=[256,256]):
    px = ((px*resize[1])-pos_x)/det_scale
    py = ((py*resize[0])-pos_y)/det_scale
    
    return [px,py]

def get_new_point_resize_blaze(px,py,scale_x,scale_y,resize=[256,256]):

    px = (px*resize[1])/scale_x
    py = (py*resize[0])/scale_y
    
    return [px,py]

def get_new_bbox_pad(bbox,pos_x,pos_y,det_scale):
    x1,y1,x2,y2 = bbox
    bbox[0] = (x1-pos_x)/det_scale
    bbox[1] = (y1-pos_y)/det_scale
    bbox[2] = (x2-pos_x)/det_scale
    bbox[3] = (y2-pos_y)/det_scale

    bbox = bbox.astype(np.float32)
    #bbox = bbox.astype(np.int)

    return bbox

def get_new_bbox_resize(det,scale_x,scale_y):

    x1,y1,x2,y2 = det

    det[0] = x1/scale_x
    det[1] = y1/scale_y
    det[2] = x2/scale_x
    det[3] = y2/scale_y

    det = det.astype(np.float32)
    #det = det.astype(np.int)

    return det

def get_new_kps_pad(kps,pos_x,pos_y,det_scale):
    new_kps = []
    for ki,kp in enumerate(kps):
        new_kp = [((kp[0]-pos_x)/det_scale),((kp[1]-pos_y)/det_scale)]
        new_kps.append(new_kp)

    new_kps = np.array(new_kps,dtype=np.float32)
    
    return new_kps

def get_new_kps_pad_retina(kps,pos_x,pos_y,det_scale):
    new_kps = []
    for ki in range(0,len(kps),2):
        new_kp = [((kps[ki]-pos_x)/det_scale),((kps[ki+1]-pos_y)/det_scale)]
        new_kps.append(new_kp)

    new_kps = np.array(new_kps,dtype=np.float32)
    
    return new_kps

def get_new_kps_resize(kps,scale_x,scale_y):
    new_kps = []
    for ki in range(0,len(kps),2):
        new_kp = [(kps[ki]/scale_x),(kps[ki+1]/scale_y)]
        new_kps.append(new_kp)
    new_kps = np.array(new_kps,dtype=np.float32)
    
    return new_kps

def get_new_kps_resize_insight(kps,scale_x,scale_y):
    new_kps = []
    for ki,kp in enumerate(kps):
        new_kp = [(kp[0]/scale_x),(kp[1]/scale_y)]
        new_kps.append(new_kp)

    new_kps = np.array(new_kps,dtype=np.float32)
    
    return new_kps





def face_align(img,lmark_ref,kps,out_size):

    #new_kps = get_new_kps(kps,pos_x,pos_y,det_scale)
    
    st = skimage.transform.SimilarityTransform()
    st.estimate(kps, lmark_ref)
    M = st.params[0:2, :]
    
    aligned = cv2.warpAffine(img, M, (out_size,out_size), borderMode=cv2.BORDER_REPLICATE)

    return aligned