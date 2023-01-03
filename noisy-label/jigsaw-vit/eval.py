# from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb
import os
import argparse

import torch
import torch.nn as nn

import utils
import torchvision
from datasets import build_transform
import model_jigsaw


parser = argparse.ArgumentParser(description='PyTorch Jigsaw-ViT Classification Evaluation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--data-path', type=str, default='../data/Animal10N/', help='val directory')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--data-set', type=str, choices=['Animal10N', 'Clothing1M', 'Food101N'], default='Animal10N', help='which dataset?')
parser.add_argument('--batch-size', default=100, type=int, help='Batch size per GPU (effective batch size is batch_size * # gpus')

parser.add_argument('--arch', type=str, default='vit_small_patch16', choices=['vit_small_patch16', 'vit_base_patch16', 'vit_large_patch16'], help='which arch')
parser.add_argument('--resumePth', type=str, help='resume path')
parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

args = parser.parse_args()
print (args)
# -------------------------------------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# test data loading
transform = build_transform(is_train=False, args=args)

if args.data_set == 'Animal10N' :
    root = os.path.join(args.data_path, 'test')
    nb_cls = 10
elif args.data_set == 'Clothing1M' :
    root = os.path.join(args.data_path, 'clean_test')
    nb_cls = 14
elif args.data_set == 'Food101N' :
    root = os.path.join(args.data_path, 'test')
    nb_cls = 101

val_dataset = torchvision.datasets.ImageFolder(root, transform=transform)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, drop_last=False, num_workers=16)

# create model
print("=> creating model '{}'".format(args.arch))
net = model_jigsaw.create_model(args.arch, nb_cls)

if args.resumePth :
    param = torch.load(os.path.join(args.resumePth, 'netBest.pth'))
    utils.load_my_state_dict(net, param['net'])
    print ('Loading net weight from {}'.format(args.resumePth))

net.cuda()
net.eval()

print("Evaluate on {} test set".format(args.data_set))

top1 = utils.AverageMeter()

for batchIdx, (inputs, targets) in enumerate(val_loader):   
    inputs = inputs.cuda() 
    targets = targets.cuda()

    with torch.no_grad():
        outputs= net.forward_cls(inputs)
    _, pred = torch.max(outputs, dim=1)

    acc1 = utils.accuracy(outputs, targets, topk=(1,))
    top1.update(acc1[0].item(), inputs.size()[0])

    msg = 'Top1: {:.3f}%'.format(top1.avg)
    utils.progress_bar(batchIdx, len(val_loader), msg)

msg = 'Test accuracy: {:.3f}%'.format(top1.avg)
print(msg)
