# from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb
import numpy as np
import json
import os
import argparse
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import datasets
import torchvision.transforms as transforms

from timm.models.vision_transformer import vit_small_patch16_224

from torchattacks import *

from datasets import build_transform
from torch.nn import Parameter
import utils

parser = argparse.ArgumentParser(description='PyTorch Classification + MAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--data-path', type=str, default='./data/imagenet/', help='val directory')
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--attack-type', type=str, choices=['FGSM', 'BIM', 'PGD', 'MI', 'AutoAttack', 'Square', 'CW'], default='FGSM', help='which attack?')
parser.add_argument('--eps', type=float, default=8/255, help='epsilon')
parser.add_argument('--steps', type=int, default=10, help='steps')
parser.add_argument('--n-queries', type=int, default=100, help='steps')

parser.add_argument('--resumePth', type=str, help='resume path1')
parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

args = parser.parse_args()
print (args)
# -------------------------------------------------------------------------------------------------------
# eval transform
transform_val = build_transform(is_train=False, args=args)

if args.attack_type == 'AutoAttack' :
    root = os.path.join(args.data_path, 'val_rand')
else:
    root = os.path.join(args.data_path, 'val')
imagnet_data = datasets.ImageFolder(root, transform=transform_val)

data_loader = torch.utils.data.DataLoader(imagnet_data, 
                                          batch_size=1, 
                                          shuffle=False)

# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
# Model
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = vit_small_patch16_224(pretrained=False)
if args.resumePth :
    param = torch.load(args.resumePth)
    utils.load_my_state_dict(model, param['model'])
    print ('Loading net weight from {}'.format(args.resumePth))

total_model = nn.Sequential(
    norm_layer,
    model
).cuda()

total_model = total_model.eval()

# -------------------------------------------------------------------------------------------------------

print("-"*70)
print(args.attack_type)

if args.attack_type == 'FGSM' :
    atk = FGSM(total_model, eps=args.eps)
elif args.attack_type == 'PGD' :
    atk = PGD(total_model, eps=args.eps, alpha=2/225, steps=args.steps, random_start=True)
elif args.attack_type == 'Square' :
    atk = Square(total_model, eps=args.eps, n_queries=args.n_queries, n_restarts=1, loss='ce')
elif args.attack_type == 'CW' :
    atk = CW(total_model, c=1, lr=0.01, steps=args.steps, kappa=0)

print("Adversarial Image & Predicted Label")


top1 = utils.AverageMeter()
top5 = utils.AverageMeter()

for batchIdx, (images, labels) in enumerate(data_loader):   

    start = time.time()
    adv_images = atk(images, labels)
    labels = labels.cuda()

    outputs = total_model(adv_images.cuda())

    _, pre = torch.max(outputs.data, 1)

    acc1, acc5 = utils.accuracy(outputs.data, labels, topk=(1,5))

    top1.update(acc1[0].item(), images.size()[0])
    top5.update(acc5[0].item(), images.size()[0])

    msg = 'Top1: {:.3f}%, Top5: {:.3f}%'.format(top1.avg, top5.avg)
    utils.progress_bar(batchIdx, len(data_loader), msg)

print('Total elapsed time (sec): %.2f' % (time.time() - start))
print('Robust accuracy Top1: %.2f %%, Top5: %.2f %%' % (top1.avg, top5.avg))
