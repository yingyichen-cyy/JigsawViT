#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils
import model_jigsaw
from datasets import build_dataset

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


### ----------------------------------- Weighted training ---------------------------------- ###
def weighted_training(inputs,
                      targets,
                      inputs_mixup,
                      targets_mixup, 
                      iter_count,
                      niter_eval,
                      mask_ratio,
                      net,
                      criterion_sup,
                      criterion_unsup,
                      eta,
                      losses_jigsaw,
                      losses_cls,
                      top1,
                      top1_jigsaw,
                      args):

    outputs_jigsaw, targets_jigsaw, outputs_cls = net(inputs, inputs_mixup, mask_ratio)

    loss_jigsaw = criterion_unsup(outputs_jigsaw, targets_jigsaw)
    if args.rank == 0 : 
        losses_jigsaw.update(loss_jigsaw.item(), inputs.size()[0])
    acc_jigsaw = utils.accuracy(outputs_jigsaw, targets_jigsaw, topk=(1,))
    if args.rank == 0 : 
        top1_jigsaw.update(acc_jigsaw[0].item(), inputs.size()[0])

    loss_cls = criterion_sup(outputs_cls, targets_mixup)
    if args.rank == 0 : 
        losses_cls.update(loss_cls.item(), inputs_mixup.size()[0])
    acc_cls = utils.accuracy(outputs_cls, targets, topk=(1,))
    if args.rank == 0 : 
        top1.update(acc_cls[0].item(), inputs_mixup.size()[0])
    
    loss = loss_cls + loss_jigsaw * eta
    return loss, losses_jigsaw, losses_cls, top1, top1_jigsaw

### --------------------------------------------------------------------------------------------


### ---------------------------------------- Train ----------------------------------------- ###
def Train(args,
          train_sampler,
          history,
          optimizer,
          train_loader,
          val_loader,
          niter_eval,
          net,
          criterion_sup,
          criterion_unsup,
          mask_ratio,
          warmup_iter,
          total_iter,
          max_lr,
          min_lr,
          out_dir,
          logger,
          training_func,
          mixup_fn,
          eta) :
    net.train()
    
    best_acc = 0  # best test accuracy
    # Log   
    losses_jigsaw = utils.AverageMeter()
    losses_cls = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_jigsaw = utils.AverageMeter()
    
    lr_log = utils.AverageMeter()
    
    iter_count = 0 
    epoch = 0
    
    while True: 
        epoch += 1
        if args.distributed :
            train_sampler.set_epoch(epoch)

        for batchIdx, (inputs, targets) in enumerate(train_loader) :

            if args.gpu is not None :
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

            if mixup_fn is not None :
                inputs_mixup, targets_mixup = mixup_fn(inputs, targets)

            lr_current = utils.adjust_learning_rate(optimizer, iter_count + 1, warmup_iter, total_iter, max_lr, min_lr)
            
            optimizer.zero_grad()
    
            loss, losses_jigsaw, losses_cls, top1, top1_jigsaw = training_func(inputs, targets, inputs_mixup, targets_mixup, iter_count, niter_eval, mask_ratio, net, criterion_sup, criterion_unsup, eta, losses_jigsaw, losses_cls, top1, top1_jigsaw, args)
            loss.backward()
            optimizer.step()
            if args.rank == 0 :
                lr_log.update(lr_current, 1)
            
            if iter_count % args.print_every == args.print_every - 1 and args.rank == 0  :
                msg = 'Training Iter: {:d} / {:d} | Lr: {:.7f} | Loss_Jigsaw: {:.3f} | Loss_cls: {:.3f} | Top1 Jigsaw: {:.3f}% | Top1 Cls: {:.3f}%'.format(iter_count, total_iter, lr_log.avg, losses_jigsaw.avg, losses_cls.avg, top1_jigsaw.avg, top1.avg)
                logger.info(msg)
                    
            if iter_count % niter_eval == niter_eval - 1 and args.rank == 0: 
                with torch.no_grad() :
                    best_acc, acc = Test(args, iter_count, best_acc, val_loader, net, out_dir, logger)

                history['iter'].append(iter_count)
                history['trainLoss_jigsaw'].append(losses_jigsaw.avg)
                history['trainLoss_cls'].append(losses_cls.avg)
                history['trainTop1'].append(top1.avg)
                history['trainTop1_jigsaw'].append(top1_jigsaw.avg)

                history['testTop1'].append(acc)      
                history['best_acc'].append(best_acc)

                with open(os.path.join(out_dir, 'history.json'), 'w') as f: 
                    json.dump(history, f)

                # Log   
                losses_jigsaw = utils.AverageMeter()
                losses_cls = utils.AverageMeter()
                top1 = utils.AverageMeter()
                top1_jigsaw = utils.AverageMeter()

                lr_log = utils.AverageMeter()
                net.train()
            iter_count += 1 
            if iter_count > total_iter : 
                return best_acc
        
### --------------------------------------------------------------------------------------------          


### ---------------------------------------- Test ------------------------------------------ ###
def Test(args, iter_count, best_acc, val_loader, net, out_dir, logger) :
    net.eval()

    bestTop1 = 0  
    true_pred = torch.zeros(1).cuda()
    nb_sample = 0

    for batchIdx, (inputs, targets) in enumerate(val_loader) :
        inputs = inputs.cuda() 
        targets = targets.cuda()

        _, _, outputs= net(inputs, inputs, args.mask_ratio)
        _, pred = torch.max(outputs, dim=1)

        true_pred = true_pred + torch.sum(pred == targets).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

    acc = true_pred / nb_sample
    acc = acc.item()

    if args.rank == 0 :
        msg = 'Test Iter {:d}, Acc {:.3f} %,  (Best Acc {:.3f} %)'.format(iter_count, acc * 100, best_acc * 100)
        logger.info(msg)

    # save checkpoint
    if acc > best_acc and args.rank == 0:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc * 100, acc * 100)
        logger.info(msg)
        logger.info('Saving Best!!!')
        param = {'net': net.state_dict()}
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))

        best_acc = acc

    return best_acc, acc

### --------------------------------------------------------------------------------------------          

 
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------                         
#----------------------------------------------------------------------------------------------- 

def main_worker(gpu, ngpus_per_node, args) :
    args.gpu = gpu
    


    # suppress printing if not master
    if args.multiprocessing_distributed and gpu != 0 :
        def print_pass(*args) :
            pass
        builtins.print = print_pass

    if args.gpu is not None :
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed :
        if args.dist_url == "env://" and args.rank == -1 :
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed :
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    train_dataset, nb_cls = build_dataset(is_train=True, args=args)
    val_dataset, _ = build_dataset(is_train=False, args=args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    net = model_jigsaw.create_model(args.arch, nb_cls)

    if args.distributed :
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None :
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        else :
            net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None :
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else :
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=nb_cls)

    # criterion for supervised learning loss
    if mixup_fn is not None :
        # smoothing is handled with mixup label transform
        criterion_sup = SoftTargetCrossEntropy().cuda(args.gpu)
    elif args.smoothing > 0. :
        criterion_sup = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda(args.gpu)
    else :
        criterion_sup = nn.CrossEntropyLoss().cuda(args.gpu)

    # criterion for unsupervised learning loss
    criterion_unsup = nn.CrossEntropyLoss().cuda(args.gpu)

    print("criterion_sup = %s" % str(criterion_sup))

    optimizer = torch.optim.AdamW(net.parameters(), 
                                  lr = args.min_lr, 
                                  betas = (0.9, 0.95), 
                                  weight_decay = args.weight_decay)

    # optionally resume from a checkpoint
    if args.resumePth :
        if os.path.isfile(args.resumePth) :
            print("=> loading checkpoint '{}'".format(args.resumePth))
            if args.gpu is None :
                checkpoint = torch.load(args.resumePth)
            else :
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resumePth, map_location=loc)
            # net.load_state_dict(checkpoint['net'])
            utils.load_my_state_dict(net, checkpoint)
            print("=> loaded checkpoint '{}'".format(args.resumePth))
        else :
            print("=> no checkpoint found at '{}'".format(args.resumePth))

    cudnn.benchmark = True

    # Data loading code
    if args.distributed :
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        print("train_sampler = %s" % str(train_sampler))
        if args.dist_eval :
            val_sampler = torch.utils.data.DistributedSampler(val_dataset)
        else :
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    else :
        train_sampler = None
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, 
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
  
    train_func = weighted_training
    
    logger = utils.get_logger(args.out_dir)
    if args.rank == 0 :
        logger.info(args)
    history = {'iter' : [], 'trainTop1':[], 'trainTop1_jigsaw':[], 'best_acc':[], 'testTop1':[], 'trainLoss_cls':[], 'trainLoss_jigsaw':[]}
            
    best_acc = Train(args,
                     train_sampler,
                     history,
                     optimizer,
                     train_loader,
                     val_loader,
                     args.niter_eval,
                     net,
                     criterion_sup,
                     criterion_unsup,
                     args.mask_ratio,
                     args.warmup_iter,
                     args.total_iter,
                     args.max_lr,
                     args.min_lr,
                     args.out_dir,
                     logger,
                     train_func,
                     mixup_fn,
                     args.eta)
        
    msg = 'mv {} {}'.format(args.out_dir, '{}_Acc{:.3f}'.format(args.out_dir, best_acc))
    logger.info(msg)
    os.system(msg)


def main() :
    args = parser.parse_args()
    # output dir + loss + optimizer    
    if not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



if __name__ == '__main__': 
                        
    parser = argparse.ArgumentParser(description='PyTorch Classification + MAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--data-path', type=str, default='../../data/Animal10N/', help='val directory')
    parser.add_argument('--data-set', type=str, choices=['CIFAR10', 'Animal10N', 'Clothing1M', 'Food101N'], default='CIFAR10', help='which dataset?')

    parser.add_argument('--arch', type=str, default='vit_small_patch16', choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'], help='which arch')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--workers', type=int, default=16, help='nb of workers')
    
    parser.add_argument('--max-lr', type=float, default=1e-3, help='max learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='min learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--mask-ratio', type=float, default=0.75, help='mask ratio')
    parser.add_argument('--total-iter', type=int, default=20000, help='training iterations')
    parser.add_argument('--warmup-iter', type=int, default=2000, help='warmup iterations')
    
    parser.add_argument('--niter-eval', type=int, default=1000, help='nb of iterations for evaluation')
    parser.add_argument('--print-every', type=int, default=100, help='print training info after how many iterations')
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--resumePth', type=str, help='resume path')
    parser.add_argument('--gpu', type=str, default=None, help='gpu devices')
    parser.add_argument('--eta', type=float, default=1, help='eta, weight for the reconstruction loss')
    
    # Augmentation parameters (reference: MAE)
    parser.add_argument('--input-size', default=64, type=int, help='images input size')
    parser.add_argument('--color-jitter', type=float, default=None, help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Random Erase params (reference: MAE)
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # * Mixup params (reference: MAE)
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distrbuted learning
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    main()