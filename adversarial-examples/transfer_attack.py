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
import imagenet_resnet as resnet


parser = argparse.ArgumentParser(description='PyTorch Classification + MAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--data-path', type=str, default='./data/imagenet/', help='val directory')
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--arch', type=str, choices=['resnet152', 'vit'], default='vit', help='which archtecture?')
parser.add_argument('--attack-type', type=str, choices=['FGSM', 'BIM', 'PGD', 'MI', 'AutoAttack'], default='FGSM', help='which attack?')
parser.add_argument('--eps', type=float, default=16/255, help='epsilon')
parser.add_argument('--steps', type=int, default=10, help='steps')

parser.add_argument('--resumePth1', type=str, help='resume path1')
parser.add_argument('--resumePth2', type=str, help='resume path2')
parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

args = parser.parse_args()
print (args)
# -------------------------------------------------------------------------------------------------------
# eval transform
transform_val = build_transform(is_train=False, args=args)

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

if args.arch == 'vit' :
    model = vit_small_patch16_224(pretrained=True)
elif args.arch == 'resnet152' :
    model = resnet.resnet152(pretrained=True)


model1 = vit_small_patch16_224(pretrained=False)
if args.resumePth1 :
    param = torch.load(args.resumePth1)
    utils.load_my_state_dict(model1, param['model'])
    print ('Loading net weight from {}'.format(args.resumePth1))

model2 = vit_small_patch16_224(pretrained=False)
if args.resumePth2 :
    param = torch.load(args.resumePth2)
    utils.load_my_state_dict(model2, param['model'])
    print ('Loading net weight from {}'.format(args.resumePth2))

total_model = nn.Sequential(
    norm_layer,
    model
).cuda()

total_model = total_model.eval()


total_model1 = nn.Sequential(
    norm_layer,
    model1
).cuda()

total_model1 = total_model1.eval()


total_model2 = nn.Sequential(
    norm_layer,
    model2
).cuda()

total_model2 = total_model2.eval()

# -------------------------------------------------------------------------------------------------------

print("-"*70)
print(args.attack_type)

if args.attack_type == 'FGSM' :
    atk = FGSM(total_model, eps=args.eps)
elif args.attack_type == 'BIM' :
    atk = BIM(total_model, eps=args.eps, alpha=2/255, steps=args.steps)
elif args.attack_type == 'PGD' :
    atk = PGD(total_model, eps=args.eps, alpha=2/225, steps=args.steps, random_start=True)
elif args.attack_type == 'MI' :
    atk = MIFGSM(total_model, eps=args.eps, alpha=2/255, steps=args.steps, decay=0.1)
elif args.attack_type == 'AutoAttack' :
    atk = AutoAttack(total_model, norm='Linf', eps=args.eps, version='standard', n_classes=1000)
    # atk = AutoAttack(model, norm='L2', eps=args.eps, version='standard', n_classes=1000)


print("Adversarial Image & Predicted Label")


top1 = utils.AverageMeter()
top1_deit = utils.AverageMeter()
top1_jigsaw = utils.AverageMeter()

top5 = utils.AverageMeter()
top5_deit = utils.AverageMeter()
top5_jigsaw = utils.AverageMeter()

for batchIdx, (images, labels) in enumerate(data_loader):   

    start = time.time()
    adv_images = atk(images, labels)
    labels = labels.cuda()

    outputs = total_model(adv_images.cuda())
    outputs1 = total_model1(adv_images.cuda())
    outputs2 = total_model2(adv_images.cuda())

    _, pre = torch.max(outputs.data, 1)
    _, pre1 = torch.max(outputs1.data, 1)
    _, pre2 = torch.max(outputs2.data, 1)

    acc1, acc5 = utils.accuracy(outputs.data, labels, topk=(1,5))
    acc1_1, acc5_1 = utils.accuracy(outputs1.data, labels, topk=(1,5))
    acc1_2, acc5_2 = utils.accuracy(outputs2.data, labels, topk=(1,5))

    top1.update(acc1[0].item(), images.size()[0])
    top1_deit.update(acc1_1[0].item(), images.size()[0])
    top1_jigsaw.update(acc1_2[0].item(), images.size()[0])

    top5.update(acc5[0].item(), images.size()[0])
    top5_deit.update(acc5_1[0].item(), images.size()[0])
    top5_jigsaw.update(acc5_2[0].item(), images.size()[0])

    msg = 'Top1: {:.3f}%, Top5: {:.3f}% | DeiT Top1: {:.3f}%, Top5: {:.3f}% | Jigsaw-ViT Top1: {:.3f}, Top5: {:.3f}%'.format(top1.avg, top5.avg, top1_deit.avg, top5_deit.avg, top1_jigsaw.avg, top5_jigsaw.avg)
    utils.progress_bar(batchIdx, len(data_loader), msg)

print('Total elapsed time (sec): %.2f' % (time.time() - start))
print('Robust accuracy: %.2f %% | Model1 Top1: %.2f %%, Top5: %.2f %% | Model2 Top1: %.2f %%, Top5: %.2f %%' % (top1.avg, top1_deit.avg, top5_deit.avg, top1_jigsaw.avg, top5_jigsaw.avg))
