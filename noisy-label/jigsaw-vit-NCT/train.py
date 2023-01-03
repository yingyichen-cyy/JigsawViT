# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from datetime import datetime

import loss
import utils
import numpy as np
import os
import json


def GaussianDist(mu, std, N):
    dist = np.array([np.exp(-((i - mu) / std)**2) for i in range(1, N + 1)])
    return dist / np.sum(dist)


# separate the data batch into two 
# for the two co-teaching networks
def groupClsData(dataCls, device): 
    imagesCls, labelsCls = dataCls['Iorg'].to(device), dataCls['Corg'].to(device)
    
    bs = imagesCls.size()[0]
    cls1 = imagesCls.narrow(0, 0, bs //2)
    label1 = labelsCls.narrow(0, 0, bs //2)
    
    cls2 = imagesCls.narrow(0, bs //2, bs //2)
    label2 = labelsCls.narrow(0, bs //2, bs //2)
    
    return cls1, cls2, label1, label2

def switch_to_train_mode(net1, net2): 
    net1.train()
    net2.train() 

def switch_to_eval_mode(net1, net2): 
    net1.eval()   
    net2.eval() 

def Train(history,
          best_acc,
          optimizer1, 
          optimizer2, 
          trainLoaderCls, 
          valLoaderCls,
          niter_eval,
          net1,
          net2,
          criterion_sup, 
          criterion_unsup,
          mask_ratio, 
          warmup_iter,
          total_iter,
          max_lr,
          min_lr,
          out_dir,
          logger,
          mixup_fn, 
          eta, 
          rateSchedule,
          device,
          dist,
          mask_feat_dim) : 

    lossTotal1 = utils.AverageMeter()
    lossTotal2 = utils.AverageMeter()

    lr_log = utils.AverageMeter()

    acc1 = utils.AverageMeter()
    acc2 = utils.AverageMeter()

    iter_count = 0 

    while True :

        for batchIdx, dataCls in enumerate(trainLoaderCls):
        
            imagesCls1, imagesCls2, labelsCls1, labelsCls2 = groupClsData(dataCls, device)

            if mixup_fn is not None :
                imagesCls1_mixup, labelsCls1_mixup = mixup_fn(imagesCls1, labelsCls1)
                imagesCls2_mixup, labelsCls2_mixup = mixup_fn(imagesCls2, labelsCls2)

            switch_to_eval_mode(net1, net2)

            forgetRate = rateSchedule[iter_count]

            with torch.no_grad() : 
                logitsCls1, _ = net1.forward_cls(imagesCls1, k=history['valK'][-1])
                logitsCls2, _ = net2.forward_cls(imagesCls2, k=history['valK'][-1])
                idx1Final, idx2Final, nbRemember = loss.SampleSelection(logitsCls1, logitsCls2, labelsCls1, labelsCls2, forgetRate)
        
            acc1Batch = utils.accuracy(logitsCls1, labelsCls1, topk=(1,))
            acc2Batch = utils.accuracy(logitsCls2, labelsCls2, topk=(1,))

            lr_current1 = utils.adjust_learning_rate(optimizer1, iter_count + 1, warmup_iter, total_iter, max_lr, min_lr)
            lr_current2 = utils.adjust_learning_rate(optimizer2, iter_count + 1, warmup_iter, total_iter, max_lr, min_lr)

            switch_to_train_mode(net1, net2)
            loss1, loss2 = loss.Classification(eta, criterion_sup, criterion_unsup, mask_ratio, imagesCls1, imagesCls1_mixup, imagesCls2, imagesCls2_mixup, labelsCls1, labelsCls1_mixup, labelsCls2, labelsCls2_mixup, idx1Final, idx2Final, net1, net2, dist)
        
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
        
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            lr_log.update(lr_current2, 1)

            lossTotal1.update(loss1.item(), nbRemember)
            lossTotal2.update(loss2.item(), nbRemember)
        
            acc1.update(acc1Batch[0].item(), imagesCls1.size()[0])
            acc2.update(acc2Batch[0].item(), imagesCls2.size()[0])
            
            if iter_count % niter_eval == niter_eval - 1 : 

                msg = 'Training Iter: {:d} / {:d} | Lr: {:.7f} | Loss_total1: {:.3f} | Loss_total2: {:.3f} | Top1_Cls1: {:.3f}% | Top1_Cls2: {:.3f}%'.format(iter_count, total_iter, lr_log.avg, lossTotal1.avg, lossTotal2.avg, acc1.avg, acc2.avg)
                logger.info(msg)
                with torch.no_grad() :
                    best_acc, acc, history = Test(iter_count, best_acc, valLoaderCls, net1, net2, device, history, out_dir, logger, dist, mask_feat_dim)
                
                history['iter'].append(iter_count)
                history['trainLoss_total1'].append(lossTotal1.avg)
                history['trainLoss_total2'].append(lossTotal2.avg)
                history['trainAcc1'].append(acc1.avg)
                history['trainAcc2'].append(acc2.avg)
                
                history['testTop1'].append(acc)      
                history['best_acc'].append(best_acc)

                with open(os.path.join(out_dir, 'history.json'), 'w') as f: 
                    json.dump(history, f)

                # Log   
                lossTotal1 = utils.AverageMeter()
                lossTotal2 = utils.AverageMeter()
                acc1 = utils.AverageMeter()
                acc2 = utils.AverageMeter()

                lr_log = utils.AverageMeter()

            iter_count += 1 
            if iter_count > total_iter : 
                return best_acc
                

def Test(iter_count, best_acc, valLoaderCls, net1, net2, device, history, out_dir, logger, dist, mask_feat_dim):   
    switch_to_eval_mode(net1, net2)

    bestTop1 = 0   
    true_pred = torch.zeros(1).cuda()
    nb_sample = 0
            
    for batchIdx, dataCls in enumerate(valLoaderCls):
        imagesCls, labelsCls = dataCls['Iorg'].to(device), dataCls['Corg'].to(device)

        _, feature1 = net1.forward_cls(imagesCls)
        _, feature2 = net2.forward_cls(imagesCls)
        outputs = []

        if dist is not None:
            for i in range(len(mask_feat_dim)):
                logitsCls1 = net1.head(feature1 * mask_feat_dim[i])
                logitsCls2 = net2.head(feature2 * mask_feat_dim[i])
                logitsCls = (logitsCls1 + logitsCls2) * 0.5
                outputs.append(logitsCls.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)

        _, pred = torch.max(outputs, dim=2)
        true_pred = true_pred + torch.sum(pred == labelsCls, dim=1).float()
        nb_sample += imagesCls.size(0)

    acc, k = torch.max((true_pred / nb_sample - 1e-5 * torch.arange(len(mask_feat_dim)).type_as(true_pred)), dim=0)
    acc, k = acc.item(), k.item()

    msg = 'Test Iter {:d}, Acc {:.3f} %,  (Best Acc {:.3f} %)'.format(iter_count, acc * 100, best_acc * 100)
    logger.info(msg)

    history['valAccClsTotal'].append(acc)
    history['valK'].append(k)

    if acc > best_acc :
        msg = 'Co-teaching: best Performance improved from {:.3f} --> {:.3f}'.format(best_acc * 100, acc * 100)
        logger.info(msg)
        logger.info('Saving Best!!!')
        param1 = {'net': net1.state_dict()}
        torch.save(param1, os.path.join(out_dir, 'netBest1.pth'))
        param2 = {'net': net2.state_dict()}
        torch.save(param2, os.path.join(out_dir, 'netBest2.pth'))

        best_acc = acc

    return best_acc, acc, history
