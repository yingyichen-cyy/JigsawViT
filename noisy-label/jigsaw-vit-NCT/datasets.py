# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import PIL.Image as Image

from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def LoadImg(path):
    return Image.open(path).convert('RGB')
 

class ImageFolder(Dataset):
    def __init__(self, imgDir, dataTransform, isRot=False):
        self.imgDir = imgDir
        
        self.clsList = sorted(os.listdir(imgDir))
        self.nbCls = len(self.clsList)
        self.cls2Idx = dict(zip(self.clsList, range(self.nbCls)))
        self.imgPth = []
        self.imgLabel = []
        for cls in self.clsList: 
            imgList = sorted(os.listdir(os.path.join(self.imgDir, cls)))
            self.imgPth = self.imgPth + [os.path.join(self.imgDir, cls, img) for img in imgList]
            self.imgLabel = self.imgLabel + [self.cls2Idx[cls] for _ in range(len(imgList))]
        
        
        self.nbImg = len(self.imgPth)
        self.dataTransform = dataTransform
        # whether to rotate the image
        self.isRot = isRot
        
        self.angle = {0:0, 1:90, 2:180, 3:270}
        self.angleCls = [0, 1, 2, 3]
        
    def __getitem__(self, idx):   
        I = LoadImg(self.imgPth[idx])
        
        Iorg = self.dataTransform(I)
        Corg = self.imgLabel[idx]
        
        if self.isRot: 
            Crot1, Crot2 = np.random.choice(self.angleCls, 2, replace=False)
            
            Irot1 = self.dataTransform(I.rotate(angle=self.angle[Crot1], resample=Image.BILINEAR))
            Irot2 = self.dataTransform(I.rotate(angle=self.angle[Crot2], resample=Image.BILINEAR))
            
            return {'Irot1': Irot1, 'Crot1': Crot1, 'Irot2': Irot2, 'Crot2': Crot2}
        
        else:    
            return {'Iorg': Iorg, 'Corg': Corg}
            
    def __len__(self):
        return self.nbImg


def build_dataset(is_train, args) :
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10' :
        root = os.path.join(args.data_path, 'train_sn_0.2' if is_train else 'test')
        nb_cls = 10
    elif args.data_set == 'CIFAR100' :
        root = os.path.join(args.data_path, 'train_sn_0.2' if is_train else 'test')
        nb_cls = 100
    elif args.data_set == 'Animal10N' :
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 10
    elif args.data_set == 'Clothing1M' :
        # we use a randomly selected balanced training subset
        root = os.path.join(args.data_path, 'noisy_rand_subtrain' if is_train else 'clean_val')
        nb_cls = 14
    elif args.data_set == 'Food101N' :
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 101

    dataset = ImageFolder(root, transform)

    print(dataset)

    return dataset, nb_cls


def build_transform(is_train, args) :
    if args.data_set == 'CIFAR10' or args.data_set == 'CIFAR100' :
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else :
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    resize_im = args.input_size > 32
    if is_train :
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im :
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224 :
        crop_pct = 224 / 256
    else :
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)