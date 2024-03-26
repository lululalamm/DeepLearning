from .iresnet_forhailo import iresnet_classification
from .iresnet_agengender import iresnet_agengender,iresnet_agengender_test
#from .iresnet_expression import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200,iresnet_expression
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200,iresnet50_cmt,iresnet50_cmt_16, iresnet50_cmt_res100,iresnet50_cmt_c20
from .iresnet import iresnet50_c3ae, iresnet50_cmt_c14, iresnet50_hse
from .mobilefacenet import get_mbf, get_mbf_cmt
from .groupface import groupface50, groupface100, groupface_agengender






def get_model(name, **kwargs):
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
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)
    elif name == "g50":
        return groupface50(False, **kwargs)
    elif name == "g100":
        return groupface100(False, **kwargs)
    else:
        raise ValueError()

def get_expression_model(pretrained_path,num_features):
    return iresnet_expression(pretrained_path,num_features)

def get_classification_model(pretrained_path,num_features):
    return iresnet_classification(pretrained_path,num_features)

def get_agengeder_model(pretrained_path,num_features,freeze):
    return iresnet_agengender(pretrained_path,num_features,freeze)

def get_agengeder_model_test(pretrained_path,num_features):
    return iresnet_agengender_test(pretrained_path,num_features)

def get_cmt_model(pretrained_path,num_features,freeze=False):
    return iresnet50_cmt(pretrained_path, num_features,freeze)

def get_cmt_model_c20(pretrained_path,num_features,freeze=False):
    return iresnet50_cmt_c20(pretrained_path, num_features,freeze)

def get_cmt_model_c14(pretrained_path,num_features,freeze=False):
    return iresnet50_cmt_c14(pretrained_path, num_features,freeze)

def get_cmt_model_res100(pretrained_path,num_features):
    return iresnet50_cmt_res100(pretrained_path, num_features)

def get_cmt_model_mbf(pretrained_path,fp16,num_features):
    return get_mbf_cmt(pretrained_path,fp16,num_features)

def get_cmt_16_model(pretrained_path,num_features):
    return iresnet50_cmt_16(pretrained_path, num_features)


def get_group_model(name,pretrained_path,**kwargs):
    return groupface_agengender(name,pretrained_path,**kwargs)

def get_c3ae_model(pretrained_path,num_features,category):
    return iresnet50_c3ae(pretrained_path,num_features,category)

def get_hse_model(pretrained_path,num_features,freeze=False):
    return iresnet50_hse(pretrained_path,num_features,freeze)
