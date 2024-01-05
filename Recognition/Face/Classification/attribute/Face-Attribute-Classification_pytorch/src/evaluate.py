import os
import numpy as np
import torch
from torcheval.metrics import BinaryAccuracy,AUC


# https://github.com/enochkan/torch-metrics/blob/main/torch_metrics/classification/pr.py
class Precision:
    """
    Computes precision of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of precision score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + self.epsilon)
        return precision

class Recall:
    """
    Computes recall of the predictions with respect to the true labels.

    Args:
        y_true: Tensor of Ground truth values.
        y_pred: Tensor of Predicted values.
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`

    Returns:
        Tensor of recall score
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        true_positives = torch.sum(torch.round(torch.clip(y_pred * y_true, 0, 1)))
        actual_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (actual_positives + self.epsilon)
        return recall

def F1_score(y_pred,y_true,epsilon=1e-07):

    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred,0,1)))

    possible_positives = torch.sum(torch.round(torch.clip(y_true,0,1)))
    recall = true_positives / (possible_positives + epsilon)

    predicted_positives = torch.sum(torch.round(torch.clip(y_pred,0,1)))
    precision = true_positives / (predicted_positives + epsilon)
                           
    return 2*((precision * recall) / (precision + recall + epsilon)) 



def eval(outs,gts,criterion):
    
    precision_m = Precision()
    recall_m = Recall()
    bAcc_m = BinaryAccuracy()
    auc_m = AUC()

    loss = criterion(outs.to(torch.float32),gts.to(torch.float32))
    f1_score = F1_score(outs.to(torch.float32),gts.to(torch.float32))
    precision = precision_m(outs.to(torch.float32),gts.to(torch.float32))
    recall = recall_m(outs.to(torch.float32),gts.to(torch.float32))

    acc=[]
    auc=[]
    
    for i in range(len(outs)):
        out = outs[i]
        gt = gts[i]
        
        # Binary Accuracy
        bAcc_m.update(out, gt)
        ba = bAcc_m.compute()
        acc.append(ba.item())
        
        # AUC
        auc_m.update(out, gt)
        auc_val = auc_m.compute()
        auc.append(auc_val.item())
        
    acc = np.array(acc)
    acc = np.mean(acc)
    
    auc = np.array(auc)
    auc = np.mean(auc)
    
    return loss.item(), acc, f1_score.item(), precision.item(), recall.item(), auc   