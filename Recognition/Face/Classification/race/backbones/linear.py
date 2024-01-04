import torch
import torch.nn as nn

from backbones import get_model

class ArcfaceLinear_mbf(nn.Module):
    def __init__(self,pretrained_path, net='mbf',num_class=6, num_features=512,freeze=True,fp16=False):
        super(ArcfaceLinear_mbf, self).__init__()

        self.premodel = get_model(net, dropout=0.0,fp16=fp16, num_features=num_features)
        if pretrained_path:
            self.premodel.load_state_dict(torch.load(pretrained_path))


        self.model_layer = nn.Sequential(*list(self.premodel.layers.children()))
        self.model_convsep = nn.Sequential(*list(self.premodel.conv_sep.children())) # (1,512,7,7)
        self.model_gdc = nn.Sequential(self.premodel.features.layers[0]) # LinearBlock 
        # self.model_fc = nn.Sequential(
        #                                 nn.Conv2d(512,num_class,kernel_size=1), 
        #                                 nn.Flatten(),
        #                                 nn.Softmax() 
        #                             )
        self.model_fc = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(512, num_class, bias=False),
                                        nn.BatchNorm1d(num_class),
                                        nn.LogSoftmax(dim=1)
        )

        if freeze:
            for param in self.model_layer.parameters():
                param.requires_grad = False


        
