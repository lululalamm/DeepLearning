from easydict import EasyDict as edict

config = edict()
config.lr = 1e-4
config.network = "b0"
config.num_classes = 8
config.batch_size = 512 #64
config.epochs = 40
config.pretrain_backbone = "efficientnet_b0_rwightman-3dd342df.pth"
config.output = "./save_model_224/"

config.input_size=224
config.dataset_format = "celebA/h5/230720/celeba_224_{}.h5"

config.attribute_names = ['Beard', 'Smiling', 'Eyeglasses', 'Wearing_Lipstick', 'Wearing_Hat', 'Wearing_Earrings', 'Wearing_Necklace', 'Wearing_Necktie']