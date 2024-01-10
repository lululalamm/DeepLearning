import os
import numpy as np
import cv2
#from tqdm import tqdm

from ..utils_pose.pose_util import draw_pose


def tracking_video(vid_path,save_path, detection_model, tracker,person=False,toRGB=True,**kwargs):
    cap = cv2.VideoCapture(vid_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            track_plot = tracker.dttrack(detection_model,frame,toRGB=toRGB,person=person,return_type=1,**kwargs)
            track_plot = cv2.cvtColor(track_plot,cv2.COLOR_RGB2BGR)
            vid_writer.write(track_plot)
        else:
            break
    cap.release()
    vid_writer.release()

    print("Finish Tracking ...")
    print("Save:",save_path)
    return

def tracking_video_pose(vid_path,save_path, detection_model,pose_model, tracker,person=False,toRGB=True,**kwargs):
    cap = cv2.VideoCapture(vid_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            online_tlwhs, online_ids, online_scores,track_plot = tracker.dttrack(detection_model,frame,toRGB=toRGB,person=person,return_type=3,**kwargs)
            results = pose_model.infer(frame,online_tlwhs,online_scores,toRGB=toRGB,box_format='xywh')
            track_plot = draw_pose(track_plot,results)
            track_plot = cv2.cvtColor(track_plot,cv2.COLOR_RGB2BGR)

            vid_writer.write(track_plot)
        else:
            break
    cap.release()
    vid_writer.release()

    print("Finish Tracking ...")
    print("Save:",save_path)
    return

def blur_ellipse(img,bbox_xywh):
                 
    maskShape = (img.shape[0], img.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)
    tempImg = img.copy()
    # start the face loop
    for tlwh in bbox_xywh:
        tlwh = tlwh.astype(np.int32)
        x,y,w,h = tlwh
        #blur first so that the circle is not blurred
        tempImg[y:y+h, x:x+w] = cv2.blur(tempImg[y:y+h, x:x+w] ,(23,23))
        # create the circle in the mask and in the tempImg, notice the one in the mask is full

        center = (x+(w//2),y+(h//2))
        axesLength = (round(w/2),round(h*0.575))
        angle=0
        startAngle=0
        endAngle=360
        color = (0,255,0)
        thickness=3

        cv2.ellipse(mask , center ,axesLength, angle,startAngle,endAngle,255,-1)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img,img,mask = mask_inv)
    img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    
    return dst



