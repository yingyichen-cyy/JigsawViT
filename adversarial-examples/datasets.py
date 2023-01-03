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

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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
        # root = os.path.join(args.data_path, 'noisy_rand_subtrain' if is_train else 'clean_val')
        root = os.path.join(args.data_path, 'noisy_rand_subtrain' if is_train else 'Clothing1M_my_clean_test2')
        nb_cls = 14
    elif args.data_set == 'Food101N' :
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 101
    elif args.data_set == 'DomainNet' :
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 345
    elif args.data_set == 'cub200' :
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        nb_cls = 200
        
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset, nb_cls


def build_transform(is_train, args) :
    # if args.data_set == 'CIFAR10' or args.data_set == 'CIFAR100' :
    #     mean = (0.4914, 0.4822, 0.4465)
    #     std = (0.2023, 0.1994, 0.2010)
    # else :
    #     mean = IMAGENET_DEFAULT_MEAN
    #     std = IMAGENET_DEFAULT_STD
    # elif args.data_set == 'Clothing1M' :
    #     mean = (0.6959, 0.6537, 0.6371)
    #     std = (0.3113, 0.3192, 0.3214)

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
    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)