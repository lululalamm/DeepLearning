from easydict import EasyDict as edict

config = edict()

# dataset
config.prefix = "./train/race/FairFace/aligned_fair/"
config.train_data = "./train/race/FairFace/aligned_fair_list_train_lb6.txt"
config.val_data = "./train/race/FairFace/aligned_fair_list_val_lb6.txt"

config.loss = "celoss"
config.network = "mobilenetv3-small"
config.pretrained=True
config.load_type=2

config.num_classes = 6
config.freeze = False

config.train_batch = 128
config.val_batch = 20
config.optim_way = 'sgd' # sgd adam

if config.optim_way == 'sgd':
    config.lr = 0.005
    config.momentum = 0.9
elif config.optim_way == 'adam':
    config.lr = 0.001

config.image_size = 112
config.num_epoch = 100

config.output = f"./race_FairFace-{config.network}-{config.loss}-{config.optim_way}_nf_l2"
config.trans = True
