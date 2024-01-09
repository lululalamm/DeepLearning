COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

def get_anchors(input_size=(640,640),stride=32):
    if input_size[0]==640 and stride==32:
        anchors = [[12,16, 19,36, 40,28], [36,75, 76,55, 72,146], [142,110, 192,243, 459,401]]
    elif input_size[0]==1280 and stride==64:
        anchors = [[19,27, 44,40, 38,94 ], [96,68, 86,152, 180,137 ], [140,301, 303,264, 238,542], [436,615, 739,380, 925,792]]
    else:
        print("input size or stride error")
        anchors = []
    return anchors

    