from easydict import EasyDict as edict

config = edict()
config.lr = 1e-6
config.network = "b0"
config.pretrain_backbone = "efficientnet_b0_rwightman-3dd342df.pth"
config.output = "./save_model/"

