import os
import argparse
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import utils
import model_jigsaw_nested as model_jigsaw
from datasets import build_dataset

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from train import Train, Test, GaussianDist


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--data-path', type=str, default='../../data/Animal10N/', help='val directory')
parser.add_argument('--data-set', type=str, choices=['CIFAR10', 'CIFAR100', 'Animal10N', 'Clothing1M', 'Food101N', 'DomainNet'], default='CIFAR10', help='which dataset?')

parser.add_argument('--arch', type=str, default='vit_small_patch16', choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'], help='which arch')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--nb-worker', type=int, default=16, help='nb of workers')
    
parser.add_argument('--max-lr', type=float, default=1e-3, help='max learning rate')
parser.add_argument('--min-lr', type=float, default=1e-6, help='min learning rate')
parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay')
parser.add_argument('--mask-ratio', type=float, default=0.75, help='mask ratio')
parser.add_argument('--mask-ratio-cls', type=float, default=0.75, help='mask ratio')
parser.add_argument('--total-iter', type=int, default=20000, help='training iterations')
parser.add_argument('--warmup-iter', type=int, default=2000, help='warmup iterations')
    
parser.add_argument('--niter-eval', type=int, default=1000, help='nb of iterations for evaluation')
parser.add_argument('--out-dir', type=str, help='output directory')
parser.add_argument('--resumePthList', type=str, nargs='+', help='resume path (list) of different models (running)')
parser.add_argument('--gpu', type=str, default='0', help='gpu devices')
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

# co-teaching specific
parser.add_argument('--forgetRate', type=float, help='forget rate', default=0.2)
parser.add_argument('--Gradual-iter', type=int, default=5, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')

parser.add_argument('--nested', type = float, default=0.0, help='nested std hyperparameter')  
parser.add_argument('--emb-dim', type=int, default=384, help='nb of workers')

args = parser.parse_args()
print (args)

device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
best_acc = 0.0
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)
 
dataset_train, nb_cls = build_dataset(is_train=True, args=args)
dataset_val, _ = build_dataset(is_train=False, args=args)

trainLoaderCls = torch.utils.data.DataLoader(
                                 dataset_train, batch_size = args.batch_size, 
                                 shuffle = True, 
                                 drop_last = True, 
                                 num_workers = args.nb_worker, 
                                 pin_memory = True)

valLoaderCls = torch.utils.data.DataLoader(
                                 dataset_val, 
                                 batch_size = args.batch_size, 
                                 shuffle = False, 
                                 drop_last = False, 
                                 num_workers = args.nb_worker, 
                                 pin_memory = True)

mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    print("Mixup is activated!")
    mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=nb_cls)

net1 = model_jigsaw.create_model(args.arch, nb_cls)
net2 = model_jigsaw.create_model(args.arch, nb_cls)
net1.cuda()
net2.cuda()

# load model
if args.resumePthList: 
    pth1 = os.path.join(args.resumePthList[0], 'netBest.pth')
    param1 = torch.load(pth1)
    net1.load_state_dict(param1['net'])
    print ('Loading net1 weight from {}'.format(pth1))

    pth2 = os.path.join(args.resumePthList[1], 'netBest.pth')
    param2 = torch.load(pth2)
    net2.load_state_dict(param2['net'])
    print ('Loading net2 weight from {}'.format(pth2))

rateSchedule = np.ones(args.total_iter) * args.forgetRate
rateSchedule[:args.Gradual_iter] = np.linspace(0, args.forgetRate, args.Gradual_iter)

logger = utils.get_logger(args.out_dir)
logger.info(args)

optimizer1 = torch.optim.AdamW(net1.parameters(), 
                               lr = args.min_lr, 
                               betas = (0.9, 0.95), 
                               weight_decay = args.weight_decay)

optimizer2 = torch.optim.AdamW(net2.parameters(), 
                               lr = args.min_lr, 
                               betas = (0.9, 0.95), 
                               weight_decay = args.weight_decay)


# criterion for supervised learning loss
if mixup_fn is not None :
    # smoothing is handled with mixup label transform
    criterion_sup = SoftTargetCrossEntropy()
elif args.smoothing > 0. :
    criterion_sup = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
else :
    criterion_sup = nn.CrossEntropyLoss()

# criterion for unsupervised learning loss
criterion_unsup = nn.CrossEntropyLoss()

# distribution and test function
dist = GaussianDist(mu=0, std=args.nested, N=args.emb_dim) if args.nested > 0 else None

mask_feat_dim = []
for i in range(args.emb_dim): 
    tmp = torch.cuda.FloatTensor(1, args.emb_dim).fill_(0)
    tmp[:, : (i + 1)] = 1
    mask_feat_dim.append(tmp)

print("criterion_sup = %s" % str(criterion_sup))

history = {'iter' : [], 'trainLoss_total1':[], 'trainLoss_total2':[], 'best_acc':[], 'testTop1':[], 'trainAcc1':[], 'trainAcc2':[], 'valAccClsTotal':[], 'valK':[]}

with torch.no_grad() :
    best_acc, acc, history = Test(0, best_acc, valLoaderCls, net1, net2, device, history, args.out_dir, logger, dist, mask_feat_dim)

best_acc = Train(history,
                 best_acc,
                 optimizer1, 
                 optimizer2, 
                 trainLoaderCls, 
                 valLoaderCls,
                 args.niter_eval,
                 net1,
                 net2,
                 criterion_sup, 
                 criterion_unsup,
                 args.mask_ratio, 
                 args.warmup_iter,
                 args.total_iter,
                 args.max_lr,
                 args.min_lr,
                 args.out_dir,
                 logger,
                 mixup_fn, 
                 args.eta, 
                 rateSchedule,
                 device,
                 dist,
                 mask_feat_dim)

msg = 'mv {} {}'.format(args.out_dir, '{}_Acc{:.3f}'.format(args.out_dir, best_acc))
logger.info(msg)
os.system(msg)