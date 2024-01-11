import numpy as np
from scipy.spatial import ConvexHull
import cv2

from ..utils.util_recognition import feat_norm, match_score
from ..utils.util_landmark import recon_vers

from .get_result import get_detection, get_landmark, get_features
from ..model_zoo.model_landmark.tddfa import TDDFA3D_V2
from ..data.image import read_image

def to_pixel(point:np.ndarray)->np.ndarray:
    """ returning rounds items and return ndarray of integers """
    return np.round(point).astype(int)

def blend(img1, img2):
    """Blend img1 and img2 with img2 alpha layer and put the result in img1.
    img1.shape = h,w,3
    img2.shape = h,w,4
    """
    i1 = img1.astype(np.uint16)
    i2 = img2.astype(np.uint16)
    a2 = i2[:,:,3]
    a1 = 255 - a2
    for i in range(3):
        i1[:,:,i] = (a1*i1[:,:,i]+a2*i2[:,:,i])/255
    img1[:,:,:] = i1.astype(np.uint8)

def load_tddfa(path,tdd_format):
    model_dict={'onnx':'.onnx','trt':'.v8.trt','vino':'vino','openvino':'vino'}
    model_format = model_dict[tdd_format]

    if model_format=='vino':
        tddfa_path=[path+".xml",path+".bin"]
        tdd_format = 'openvino'
    else:
        tddfa_path = path+model_format

    tddfa_model = TDDFA3D_V2(tdd_format,tddfa_path)
    
    return tddfa_model

def get_blur(image,blur_faces,tddfa_model,image_h,image_w):

    # bluring
    param_lst, roi_box_lst = tddfa_model.get_results(image,blur_faces)
    pts = recon_vers(param_lst, roi_box_lst,dense=False)
    if not type(pts) in [tuple, list]: 
        pts = [pts]

    radius_list=[]
    mask = np.zeros((image_h, image_w, 3), dtype=np.uint8)

    for i in range(len(pts)):
        
        mask_each = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        landmarks = pts[i]
        points = landmarks[:2,:36].T
        hull = ConvexHull(points)

        vertices = points[hull.vertices]
        vertices = to_pixel(vertices)

        cv2.fillPoly(mask_each, pts=[vertices], color=(255, 255, 255))
        ret, thresh = cv2.threshold(cv2.cvtColor(mask_each, cv2.COLOR_RGB2GRAY), 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]

        x, y, w, h = cv2.boundingRect(cnt)
        r = min(h, w) // 10
        radius = r
        d = 2*r+1

        cv2.dilate(mask_each, kernel=np.ones((r, r), np.uint8), dst=mask_each)
        cv2.GaussianBlur(src=mask_each, ksize=(d, d), sigmaX=r, sigmaY=r, dst=mask_each)
        
        # draw
        ori_mask = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        cv2.max( mask,mask_each, mask)

        radius_list.append(radius)
        
    # blur
    img_blurred = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    r=0

    dimg = image.copy()

    for i in range(len(pts)):  
        r = max(r,radius_list[i])
        
        d=2*((5*r+1)//5)+1 # kernel size has to be odd # blur 강도
        sigma=2*r
        
        cv2.GaussianBlur(src=dimg, ksize=(d, d), sigmaX=sigma, sigmaY=sigma, dst=img_blurred)
        blend(dimg, np.dstack((img_blurred, mask)))

    return dimg

def face_blur_image(image,dt_name,dt_model,tddfa_model,recog_model=None,ref_image=None,score_th=1.3,dt_thresh=0.3):
    
    if type(image)==str:
        image = read_image(image)
    faces = get_detection(dt_name,dt_model,image,thresh=dt_thresh,height_min=0,input_size=(640,640))

    if not ref_image is None:
        if recog_model is None:
            print("need recog model")
            return
        faces_ref = get_detection(dt_name,dt_model,ref_image,thresh=dt_thresh,height_min=0,input_size=(640,640))[0]
        ref_feat = recog_model.get(ref_image,faces_ref)
        ref_feat = feat_norm(ref_feat)
        ref_feat = np.expand_dims(ref_feat,axis=0)

        feat_list=[]
        for face in faces:
            feat = recog_model.get(image,face)
            feat = feat_norm(feat)
            feat_list.append(feat)
        feat_list = np.array(feat_list)

        if len(feat_list)==0:
            print("feat list len: 0")
            return

        # matching
        filter_idx, scores,scores_new = match_score(ref_feat,feat_list)
        blur_faces = []
        for idx in range(len(faces)):
            if not idx in filter_idx:
                blur_faces.append(faces[idx])

    else:
        blur_faces = faces

    image_h = image.shape[0]
    image_w = image.shape[1]
    blur_image = get_blur(image,blur_faces,tddfa_model,image_h,image_w)
    
    return blur_image

    

def face_blur_video(video_path,dt_name,dt_model,tddfa_model,recog_model=None,ref_image=None,score_th=1.3,dt_thresh=0.3,save_path=''):
    
    cap = cv2.VideoCapture(video_path)

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = total_frame_count/fps

    if not save_path:
        print("need save output path")
        return
    else:
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while(cap.isOpened()):
        ret,frame = cap.read()

        if ret==False:
            break
        
        faces = get_detection(dt_name,dt_model,frame,thresh=dt_thresh,height_min=0,input_size=(640,640))

        if not ref_image is None:
            if recog_model is None:
                print("need recog model")
                return

            faces_ref = get_detection(dt_name,dt_model,ref_image,thresh=dt_thresh,height_min=0,input_size=(640,640))[0]
            ref_feat = recog_model.get(ref_image,faces_ref)
            ref_feat = feat_norm(ref_feat)
            ref_feat = np.expand_dims(ref_feat,axis=0)

            feat_list=[]
            for face in faces:
                feat = recog_model.get(frame,face)
                feat = feat_norm(feat)
                feat_list.append(feat)
            feat_list = np.array(feat_list)

            if len(feat_list)==0:
                print("feat list len: 0")
                return

            # matching
            filter_idx, scores,scores_new = match_score(ref_feat,feat_list)
            blur_faces = []
            for idx in range(len(faces)):
                if not idx in filter_idx:
                    blur_faces.append(faces[idx])

        else:
            blur_faces = faces

        
        image_h = frame.shape[0]
        image_w = frame.shape[1]
        blur_image = get_blur(frame,blur_faces,tddfa_model,image_h,image_w)

        out.write(dimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print("Save:",save_path)

    return








    


