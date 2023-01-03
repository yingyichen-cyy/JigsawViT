import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# samples selection
def SampleSelection(logitsCls1, logitsCls2, labelsCls1, labelsCls2, forgetRate):
    loss1 = F.cross_entropy(logitsCls1, labelsCls1, reduction='none')
    idx1 = torch.argsort(loss1)
    loss2 = F.cross_entropy(logitsCls2, labelsCls2, reduction='none')
    idx2 = torch.argsort((loss2))

    rememberRate = 1 - forgetRate
    nbRemember = int(rememberRate * len(idx1))

    idx1Final=idx1[:nbRemember]
    idx2Final=idx2[:nbRemember]
    
    return idx1Final, idx2Final, nbRemember


# classification loss
def Classification(eta, criterion_sup, criterion_unsup, mask_ratio, imagesCls1, imagesCls1_mixup, imagesCls2, imagesCls2_mixup, labelsCls1, labelsCls1_mixup, labelsCls2, labelsCls2_mixup, idx1Final, idx2Final, net1, net2, dist):

    # jigsaw classification
    outputs1, targets_jigsaw1 = net1.forward_jigsaw(imagesCls2[idx2Final], mask_ratio=mask_ratio)
    loss_jigsaw1 = criterion_unsup(outputs1, targets_jigsaw1)
    outputs2, targets_jigsaw2 = net2.forward_jigsaw(imagesCls1[idx1Final], mask_ratio=mask_ratio)
    loss_jigsaw2 = criterion_unsup(outputs2, targets_jigsaw2)

    # standard classification
    logitsCls1, _ = net1.forward_cls(imagesCls2_mixup[idx2Final], dist=dist)
    loss_cls1 = criterion_sup(logitsCls1, labelsCls2_mixup[idx2Final])
    logitsCls2, _ = net2.forward_cls(imagesCls1_mixup[idx1Final], dist=dist)
    loss_cls2 = criterion_sup(logitsCls2, labelsCls1_mixup[idx1Final])

    loss1 = loss_cls1 + loss_jigsaw1 * eta
    loss2 = loss_cls2 + loss_jigsaw2 * eta

    return loss1, loss2