from PIL import Image,ImageOps
import numpy as np
import os
import cv2
import torch

from ..utils.util_warp import face_align
from .constant import LMARK_REF_ARC



def read_image(img,to_bgr=False):
    if img is None:
        return "img is None"
    if type(img)==str:
        if not os.path.exists(img):
            return "img path not exists"
        img = Image.open(img)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
    img = np.array(img)
    if to_bgr:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img

def read_image_cv2(img,to_rgb=True):

    if img is None:
        return "img is None"

    if type(img)==str:
        if not os.path.exists(img):
            return "img path not exists"
        img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def read_image_retinaTorch(img):

    img_raw = cv2.imread(img, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    return img

def resize_image(image, size, keep_aspect_ratio=False):
    resized_frame = cv2.resize(image, size)
    return resized_frame

def resize_image_multi(image, target_size, max_size): # size (w,h)
    
    if image is None:
        return 'img is None'
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    resize = float(target_size) / float(im_size_min)
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)

    resized_frame = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    return resized_frame,resize

# bgr
def read_video(path):
    if not os.path.exists(path):
        return "video is not exists"
    vid = cv2.VideoCapture(path)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    return vid, video_frame_cnt, video_width, video_height, video_fps

def read_torchImage(img,out_size=None,to_bgr=False,not_norm=False):

    if type(img)==str:
        img = Image.open(img)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
    img = np.array(img)
    if out_size:
        img = cv2.resize(img, (out_size, out_size))
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    if not not_norm:
        img.div_(255).sub_(0.5).div_(0.5)

    return img

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 16), np.mod(dh, 16)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

'''
def read_retina_torch(path,resize=False,static_size=(0,0)):

    img_raw = cv2.imread(path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    if resize:
        target_size = 800
        max_size = 1200
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    if static_size[0]!=0:
        img = cv2.resize(img,static_size)

    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()

    return img
'''