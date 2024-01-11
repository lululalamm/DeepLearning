import torch.nn as nn
import torch
from collections import OrderedDict
import sys

from ..backbones.iresnet import iresnet50_cmt
from ..backbones.retinaface import RetinaFace, get_cfg

from ..backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from ..backbones.mobilefacenet import get_mbf
from ..backbones.race_linear import get_race_model
from ...data.image import read_torchImage
from ..backbones.efficientNet import get_efficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# for load retinaface
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
#


def get_cmt_model(pretrained_path,network,num_features):
    if network=='r50':
        return iresnet50_cmt(pretrained_path, num_features)

def get_race_model(pretrained_path,network,num_features,num_class):
    race_model = ArcfaceLinear_mbf(pretrained_path=pretrained_path, net =network, num_class=num_class, \
                        num_features=num_features,freeze=False,fp16=False)

    return Arace_modelc



def get_arcface(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)
    elif name == "eb0":
        num_features = kwargs.get("num_features", 512)
        return get_efficientNet('b0',num_features=num_features)
    elif name == 'ev2_s':
        num_features = kwargs.get("num_features", 512)
        return get_efficientNet('v2_s',num_features=num_features)
    else:
        raise ValueError()


class TorchModel:
    def __init__(self,model_name,model_path,**kwargs):
        
        self.model_path = model_path

        if model_name=='retinaface_torch':
            self.network = kwargs.get("network",'resnet50')
            self.torch_image = kwargs.get("torch_image",False)
            self.not_norm = kwargs.get("not_norm",False)
            self.cfg = get_cfg(self.network)

            self.net = RetinaFace(cfg=self.cfg,phase='test')
            self.net = load_model(self.net, self.model_path,False)

        elif model_name=='arcface_cmt':
            self.num_features = kwargs.get("num_features")
            self.torch_image = kwargs.get("torch_image",False)
            self.network = kwargs.get("network")
            self.not_norm = kwargs.get("not_norm",False)

            self.net = get_cmt_model("", self.network,self.num_features)
        
            load_weight = torch.load(self.model_path)

            if type(load_weight)==OrderedDict:
                try:
                    self.net.load_state_dict(load_weight)
                except:
                    new_state_dict = OrderedDict()
                    for n, v in load_weight.items():
                        name = n.replace("module.","") 
                        new_state_dict[name] = v
                    self.net.load_state_dict(new_state_dict)
            else:
                try:
                    self.net.load_state_dict(load_weight.module.state_dict())
                except:
                    self.net.load_state_dict(load_weight.state_dict())

        elif model_name=='arcface':
            self.network = kwargs.get("network",'r50')
            self.num_features = kwargs.get("num_features",512)
            self.fp16 = kwargs.get("fp16",False)
            self.torch_image = kwargs.get("torch_image",False)
            self.not_norm = kwargs.get("not_norm",False)

            self.net = get_arcface(self.network,num_features=self.num_features,fp16=self.fp16)

            load_weight = torch.load(self.model_path)

            if type(load_weight)==OrderedDict:
                try:
                    self.net.load_state_dict(load_weight)
                except:
                    new_state_dict = OrderedDict()
                    for n, v in load_weight.items():
                        name = n.replace("module.","") 
                        new_state_dict[name] = v
                    self.net.load_state_dict(new_state_dict)
            else:
                try:
                    self.net.load_state_dict(load_weight.module.state_dict())
                except:
                    self.net.load_state_dict(load_weight.state_dict())

        elif model_name=='arcface_race':
            self.num_features = kwargs.get("num_features",512)
            self.torch_image = kwargs.get("torch_image",False)
            self.network = kwargs.get("network",'mbf')
            self.not_norm = kwargs.get("not_norm",False)
            self.num_class = kwargs.get("num_class",6)

            self.net = get_race_model('',self.network,self.num_features,self.num_class)
        
            load_weight = torch.load(self.model_path)

            if type(load_weight)==OrderedDict:
                try:
                    self.net.load_state_dict(load_weight)
                except:
                    new_state_dict = OrderedDict()
                    for n, v in load_weight.items():
                        name = n.replace("module.","") 
                        new_state_dict[name] = v
                    self.net.load_state_dict(new_state_dict)
            else:
                try:
                    self.net.load_state_dict(load_weight.module.state_dict())
                except:
                    self.net.load_state_dict(load_weight.state_dict())

        self.net.to(device)
        self.net.eval() 

    def __call__(self,img,**kwargs):
        
        if not self.torch_image:
            img = read_torchImage(img,not_norm=self.not_norm)
        img = img.to(device)
        with torch.no_grad():
            outs = self.net(img)
        #outs = [outs[0].cpu().numpy(),outs[1].cpu().numpy()]

        return outs

    
        


