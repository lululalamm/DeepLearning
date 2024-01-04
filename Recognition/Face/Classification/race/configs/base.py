from easydict import EasyDict as edict

config = edict()

# dataset
config.prefix = "/data/"
config.train_data = "./train/race/aligned_race_list_train.txt"
config.val_data = "./train/race/aligned_race_list_val.txt"

config.loss = "arcface"
config.network = "mbf"
config.resume = False
config.output = f"./race-{config.network}-{config.loss}"

config.num_classes = 5
config.num_features = 512
config.fp16 = False
config.freeze = False

config.train_batch = 512
config.val_batch = 10
config.lr = 0.4
config.momentum = 0.9
config.image_size = 112
config.num_epoch = 20

