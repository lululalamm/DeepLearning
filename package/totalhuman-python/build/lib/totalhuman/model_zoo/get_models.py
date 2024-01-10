from .model_detection import yolov7, yolov5, yolox
from .model_attribute import gender
from .model_pose import vitpose

# from model_zoo.model_detection import yolov7, yolov5, yolox
# from model_zoo.model_attribute import gender
# from model_zoo.model_pose import vitpose


def get_detection_model(name,path,**kwargs):
    model=None
    if type(path)==list:
        model_format = "openvino"
    else:
        model_format = path.split(".")[-1]

    if 'yolov7' in name:
        model = yolov7.Yolov7(model_format,path,**kwargs)
    elif 'yolov5' in name:
        model = yolov5.Yolov5(model_format,path,**kwargs)
    elif 'yolox' in name:
        model = yolox.YoloX(model_format,path,**kwargs)
    
    if model is None:
        print("{} is None".format(name))
        return None
    else:
        print("{} {} loaded".format(name,model_format))
        return model


def get_attrib_model(name,path,**kwargs):
    model=None
    if type(path)==list:
            model_format = "openvino"
    else:
        model_format = path.split(".")[-1]

    if name=='gender':
        model = gender.Attrb_Gender(model_format,path,**kwargs)

    if model is None:
        print("{} is None".format(name))
        return None
    else:
        print("{} {} loaded".format(name,model_format))
        return model

def get_pose_model(name,path,**kwargs):
    model=None
    if type(path)==list:
            model_format = "openvino"
    else:
        model_format = path.split(".")[-1]

    if name=='vitpose':
        model = vitpose.VitPose(model_format,path,**kwargs)

    if model is None:
        print("{} is None".format(name))
        return None
    else:
        print("{} {} loaded".format(name,model_format))
        return model