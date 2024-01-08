exp_num = 0

dataset = 'wild_celebA'

multi_task_learning = False

evaluation = True

test_steps = None

datasets = dict(wildCelebA_root='safas/csv/',
                casia_root='safas/csv/') # modify

external = dict(train=dict(), val=dict(), test=dict())

img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229],
                    std=[0.2471, 0.2214, 0.2157])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,50], gamma=0.2)

data = dict(batch_size=3072, #256
            data_loader_workers=4,
            sampler=True, # weighted random sampler
            pin_memory=True)

epochs = dict(start_epoch=0, max_epoch=71)

resize = dict(height=128, width=128)

checkpoint = dict(snapshot_name="MobileNet3-larget-0.1_wildcelebA.pth.tar",
                  experiment_path='./logs_wildcelebA')

loss = dict(loss_type='amsoftmax',
            amsoftmax=dict(m=0.5,
                           s=1,
                           margin_type='cross_entropy',
                           label_smooth=False,
                           smoothing=0.1,
                           ratio=[1,1],
                           gamma=0),
            soft_triple=dict(cN=2, K=10, s=1, tau=.2, m=0.35))

model= dict(model_type='Mobilenet3',
            model_size = 'large',
            width_mult = 1.0,
            pretrained=True,
            embeding_dim=1280,
            imagenet_weights='./pretrained/imagenet/MobileNet3-large-0.1.pth')

aug = dict(type_aug=None,
            alpha=0.5,
            beta=0.5,
            aug_prob=0.7)

curves = dict(det_curve='det_curve_0.png',
              roc_curve='roc_curve_0.png')

dropout = dict(prob_dropout=0.1,
               classifier=0.1,
               type='bernoulli',
               mu=0.5,
               sigma=0.3)

data_parallel = dict(use_parallel=True,
                     parallel_params=dict(device_ids=[0,1,2,3,4,5,6,7], output_device=0))

RSC = dict(use_rsc=False,
           p=0.333,
           b=0.333)

test_dataset = dict(type='casia')

conv_cd = dict(theta=0)
