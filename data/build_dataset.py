import os
import time

import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode

from .dataprocess import (ClassificationDataProcessEval,
                          ClassificationDataProcessTrain)


def build_train_dataset_and_sampler(config, distributed):
    data_path = config.DATASET.ROOT
    train_split = config.DATASET.TRAIN_SPLIT
    train_crop_size = config.DATASET.CROP_SIZE
    mean = config.DATASET.MEAN
    std = config.DATASET.STD
    hflip_prob = config.DATASET.HFLIP_PROB
    auto_augment_policy = config.DATASET.AURO_AUGMENT_POLICY

    # Loading training data
    print("Loading training data")
    st = time.time()
    train_dir = os.path.join(data_path, train_split)
    preprocessing_train = ClassificationDataProcessTrain(
        crop_size=train_crop_size,
        mean=mean,
        std=std,
        hflip_prob=hflip_prob,
        auto_augment_policy=auto_augment_policy,
    )
    dataset_train = torchvision.datasets.ImageFolder(
        train_dir,
        preprocessing_train,
    )
    print("Loading time: ", time.time()-st)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train, drop_last=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)

    return dataset_train, train_sampler


def build_valid_dataset_and_sampler(config, distributed):
    data_path = config.DATASET.ROOT
    valid_split = config.DATASET.VALID_SPLIT
    valid_crop_size = config.DATASET.CROP_SIZE
    valid_resize_size = config.DATASET.RESIZE_SIZE
    mean = config.DATASET.MEAN
    std = config.DATASET.STD

    # Loading validation data
    print("Loading validation data")
    st = time.time()
    valid_dir = os.path.join(data_path, valid_split)
    preprocessing_valid = ClassificationDataProcessEval(
        crop_size=valid_crop_size,
        resize_size=valid_resize_size,
        mean=mean,
        std=std,
    )
    dataset_valid = torchvision.datasets.ImageFolder(
        valid_dir,
        preprocessing_valid,
    )
    print("Loading time: ", time.time()-st)

    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_valid)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_valid)

    return dataset_valid, test_sampler
