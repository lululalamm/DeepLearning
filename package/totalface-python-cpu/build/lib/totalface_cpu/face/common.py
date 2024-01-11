import numpy as np

class Face(dict):
    def __init__(self, d=None,**kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
            
        for k, v in d.items():
            setattr(self, k, v)
            
    def __setattr__(self,name,value):
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

