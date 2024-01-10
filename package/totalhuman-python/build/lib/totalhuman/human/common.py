import numpy as np

class Human(dict):
    def __init__(self, d=None,**kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
            
        for k, v in d.items():
            setattr(self, k, v)
            
    def __setattr__(self,name,value):
        super(Human, self).__setattr__(name, value)
        super(Human, self).__setitem__(name, value)

