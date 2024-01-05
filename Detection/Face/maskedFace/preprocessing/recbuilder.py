import pickle
import numpy as np
import os
import os.path as osp
import sys
import mxnet as mx


class RecBuilder():
    def __init__(self, path, tv='train',image_size=(112, 112)):
        self.path = path
        self.image_size = image_size
        self.widx = 0
        self.wlabel = 0
        self.max_label = -1

        self.tv = tv
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = mx.recordio.MXIndexedRecordIO(os.path.join(path, '{}.idx'.format(self.tv)), 
                                                    os.path.join(path, '{}.rec'.format(self.tv)),
                                                    'w')
        self.meta = []

    def add_image(self, img, label):
        #!!! img should be BGR!!!!

        idx = self.widx
        header = mx.recordio.IRHeader(0, label, idx, 0)

        if isinstance(img, np.ndarray):
            s = mx.recordio.pack_img(header,img,quality=95,img_fmt='.jpg')
        else:
            s = mx.recordio.pack(header, img)
        self.writer.write_idx(idx, s)
        self.widx += 1
