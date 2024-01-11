import cv2
import os
import numpy as np
import torch
from .util_recognition import compute_sim, compute_sims
from .util_landmark import recon_ver, calc_pose, plot_pose_box

def draw_result(img,faces,ga=False,race=False,to_bgr=False,color=(0,255,0),size=2):
        dimg = img.copy()
        if to_bgr:
            dimg = cv2.cvtColor(dimg,cv2.COLOR_BGR2RGB)
        for i in range(len(faces)):
            face = faces[i]
            x1,y1,x2,y2 = face.bbox.astype(np.int)
            
            cv2.rectangle(dimg,(x1,y1),(x2,y2),color,size)

            wtxt = ''
            if ga:
                if len(face.gender)<1:
                    wtxt+="{} {}".format('None','0')
                else:
                    wtxt+="{} {}".format(face.gender, face.age)

            if race:
                if len(face.race)<1:
                    wtxt+=" {}".format("None")
                else:
                    wtxt+=" {}".format(face.race)

            if wtxt:
                cv2.putText(dimg, wtxt,(x1-1, y2+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


        return dimg
    
def draw_result_land(img,faces,to_bgr=False,draw5=False,point_size=10,color=(0,255,0)):

        land_list = [ name for name in faces[0].keys() if "land" in name]

        dimg = img.copy()
        if to_bgr:
            dimg = cv2.cvtColor(dimg,cv2.COLOR_BGR2RGB)
        for i in range(len(faces)):
            face = faces[i]
            if len(land_list)==1:
                lands = face[land_list[0]]
            elif not draw5:
                lands = face.land
            else:
                lands = face.land5
            for land in lands:
                x = int(land[0])
                y = int(land[1])
            
                cv2.line(dimg,(x,y),(x,y),color,point_size)

        return dimg

def draw_result_sim(img,faces,refs,sim_thresh=0.4,ga=True,to_bgr=False,max_sim=False,all_draw=False):
        dimg = img.copy()
        font_scale=1
        if to_bgr:
            dimg = cv2.cvtColor(dimg,cv2.COLOR_BGR2RGB)
        refs_keys = list(refs.keys())
        refs_feat = list(refs.values())
        
        for i in range(len(faces)):
            face = faces[i]
            
            # compute sim
            if max_sim:
                if face.max_flag:
                    sims = compute_sims(face.feat,refs_feat)
                    max_idx = np.argmax(sims)
                    if sims[max_idx]>sim_thresh:
                        sim_pname = refs_keys[max_idx]
                        color=(0,255,0)
                else:
                    if all_draw:
                        #sim_pname='unknown'
                        sim_pname=""
                        color=(255,0,0)
                    else:
                        continue
            else:
                sims = compute_sims(face.feat,refs_feat)
                max_idx = np.argmax(sims)
                if sims[max_idx]>sim_thresh:
                    sim_pname = refs_keys[max_idx]
                    color=(0,255,0)
                else:
                    if all_draw:
                        sim_pname=''
                        color=(255,0,0)
                    else:
                        continue
            
            x1,y1,x2,y2 = face.bbox.astype(np.int)
            
            cv2.rectangle(dimg,(x1,y1),(x2,y2),color,2)
            
            if ga:
                if face.gender is not None and face.age is not None:
                    cv2.putText(dimg, "{} {}".format(face.gender, face.age),(x1-1, y2+25),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
                    if sim_pname:
                        cv2.putText(dimg, "{}".format(sim_pname),(x1-1, y2+45),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
                else:
                    cv2.putText(dimg, "{} {}".format('None', '0'), (x1-1, y2+25),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
                    if sim_pname:
                        cv2.putText(dimg, "{}".format(sim_pname), (x1-1, y2+45),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
            else:
                if sim_pname:
                    cv2.putText(dimg,sim_pname,(x1-1, y2+25),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
                

        return dimg

def torch2numpy(outs):
    if torch.is_tensor(outs):
        if outs.device.type=='cuda':
            outs = np.array(outs.cpu())
        else:
            outs = np.array(outs)
    return outs  

def draw_vispose(img,faces,param_lst,roi_box_lst,dense=False,std_size=120, \
                 color_box=(255,255,255),line_box=4,color_land=(255,255,255),point_land=5,color_vis=(40, 255, 0),line_vis=2,box_rate=0.6):

    dimg = img.copy()
    # draw bbox
    dimg = draw_result(dimg,faces,color=color_box,size=line_box)

    # draw land
    dimg = draw_result_land(dimg,faces,point_size=point_land,color=color_land)

    # draw vis-pose
    for param,roi_box in zip(param_lst,roi_box_lst):
        P,pose = calc_pose(param)
        ver = recon_ver(param,roi_box, dense, std_size)
        dimg = plot_pose_box(dimg, P, ver,color=color_vis,line_width=line_vis,box_rate=box_rate)

    return dimg

def draw_result_mask(img,faces,to_bgr=False,mask_color=(0,255,0),unmask_color=(255,0,0)):
        dimg = img.copy()
        font_scale=1
        if to_bgr:
            dimg = cv2.cvtColor(dimg,cv2.COLOR_BGR2RGB)
        
        for i in range(len(faces)):
            face = faces[i]
            
            mask_lb = np.argmax(face['mask_sf'])
            if mask_lb==0:
                color = unmask_color
                txt = 'unmask'
            else:
                color = mask_color
                txt = 'mask'
            
            x1,y1,x2,y2 = face.bbox.astype(np.int)
            cv2.rectangle(dimg,(x1,y1),(x2,y2),color,2)
            
            cv2.putText(dimg,txt,(x1-1, y2+25),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
                
        return dimg


def draw_result_liveness(img,faces,to_bgr=False,real_color=(0,255,0),fake_color=(255,0,0)):
        dimg = img.copy()
        font_scale=1
        if to_bgr:
            dimg = cv2.cvtColor(dimg,cv2.COLOR_BGR2RGB)
        
        for i in range(len(faces)):
            face = faces[i]
            
            if len(face['liveness_sf'])<1:
                continue
            else:
                liveness_lb = np.argmax(face['liveness_sf'])
                if liveness_lb==0:
                    color = real_color
                    txt = 'real'
                else:
                    color = fake_color
                    txt = 'fake'
            
                x1,y1,x2,y2 = face.bbox.astype(np.int)
                cv2.rectangle(dimg,(x1,y1),(x2,y2),color,2)
                
                cv2.putText(dimg,txt,(x1-1, y2+25),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2)
                
        return dimg

