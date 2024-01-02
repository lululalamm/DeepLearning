import math
import numpy as np
from math import pi

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.functional import linear, normalize
from partial_fc_broad import DistCrossEntropy

class BroadFaceArcFace(nn.Module):
    def __init__(
        self,
        in_features,
        queue_size=10000,
        compensate=True,
        fp16=True,
    ):
        super(BroadFaceArcFace, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.in_features = in_features
        feature_mb = torch.zeros(0, in_features)
        label_mb = torch.zeros(0, dtype=torch.int64)
        proxy_mb = torch.zeros(0, in_features)
        self.register_buffer("feature_mb", feature_mb)
        self.register_buffer("label_mb", label_mb)
        self.register_buffer("proxy_mb", proxy_mb)

        self.queue_size = queue_size
        self.compensate = compensate

        self.arcface = ArcFace()
        self.fp16=fp16
        self.dist_cross_entropy = DistCrossEntropy()

    def update(self, input, label):
        self.feature_mb = torch.cat([self.feature_mb, input.data], dim=0)
        self.label_mb = torch.cat([self.label_mb, label.data], dim=0)
        
        cat_weight = self.weight.data[label].clone()
        if cat_weight.dim()==3 and cat_weight.shape[1]==1:
            cat_weight = torch.reshape(cat_weight,(cat_weight.shape[0],cat_weight.shape[2]))
        
        self.proxy_mb = torch.cat(
            [self.proxy_mb, cat_weight], dim=0
        )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            self.proxy_mb = self.proxy_mb[over_size:]


        assert (
            self.feature_mb.shape[0] == self.label_mb.shape[0] == self.proxy_mb.shape[0]
        )

        # assert ( self.feature_mb.shape[0] == self.label_mb.shape[0] )

    def forward(self, input, label,weight,local_rank):
        # input is not l2 normalized
        self.weight = weight

        if self.compensate:
            weight_now = self.weight.data[self.label_mb]
            if weight_now.dim()==3 and weight_now.shape[1]==1:
                weight_now = torch.reshape(weight_now,(weight_now.shape[0],weight_now.shape[2]))
            delta_weight = weight_now - self.proxy_mb

            update_feature_mb = (
                self.feature_mb
                + (
                    self.feature_mb.norm(p=2, dim=1, keepdim=True)
                    / self.proxy_mb.norm(p=2, dim=1, keepdim=True)
                )
                * delta_weight
            )
        else:
            update_feature_mb = self.feature_mb
        #update_feature_mb = self.feature_mb

        large_input = torch.cat([update_feature_mb, input.data], dim=0)
        large_label = torch.cat([self.label_mb, label], dim=0)

        # modify 
        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(input)
            norm_weight_activated = normalize(self.weight.data)

            norm_embeddings_large = normalize(large_input)
            norm_weight_activated_large = normalize(self.weight)

            logits_batch = linear(norm_embeddings, norm_weight_activated)
            logits_broad = linear(norm_embeddings_large, norm_weight_activated_large)

        if self.fp16:
            logits_batch = logits_batch.float()
            logits_broad = logits_broad.float()
        logits_batch = logits_batch.clamp(-1, 1)
        logits_broad = logits_broad.clamp(-1, 1)

        logits_batch = self.arcface(logits_batch, label)
        batch_loss = self.dist_cross_entropy(logits_batch, label)
        #batch_loss = self.criterion(logits_batch,label)

        logits_broad = self.arcface(logits_broad, large_label)
        broad_loss = self.dist_cross_entropy(logits_broad, large_label)
        #broad_loss = self.criterion(logits_broad, large_label)

        # if local_rank==0:
        #     print("braod_loss:",broad_loss)
        #     print("batch_loss:",batch_loss)

        self.update(input, label)

        return batch_loss + broad_loss 