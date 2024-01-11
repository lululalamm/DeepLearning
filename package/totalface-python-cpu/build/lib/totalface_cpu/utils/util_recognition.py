from numpy.linalg import norm
import numpy as np
import torch

def compute_sim(feat1, feat2):
    
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

def compute_sims(feat1, feats):
    
    feat1 = feat1.ravel()
    sims = [ np.dot(feat1,feat2.ravel())/(norm(feat1) * norm(feat2.ravel())) for feat2 in feats]
    return sims

def feat_norm(feat):
    return feat / np.linalg.norm(feat)

def match_score(feat_src, feats,score_th=1.3):
    scores_ = np.sum(feat_src*feats,-1)
    scores_new = np.array([1+(max(-1.,min(1.,v))) for v in scores_])
    
    filter_idx = np.where(scores_new>=score_th)[0]
    scores = scores_new[filter_idx]
    
    return filter_idx, scores,scores_new