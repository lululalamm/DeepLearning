from PIL import Image,ImageOps
import numpy as np
import os
import cv2

def read_image(img,to_bgr=False):
    if type(img)==str:
        img = Image.open(img)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
    img = np.array(img)
    if to_bgr:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img