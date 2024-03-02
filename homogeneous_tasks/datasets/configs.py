import torch
import torchvision
import numpy as np

from .imagenet import ImageNet1k


datadir = './data'
imnet_dir = "./datasets/ILSVRC2012/"

cifar50 = {
    'dir': datadir,
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
}

cifar5 = {
    'dir': datadir,
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 50,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
}

imnet = {
    'dir':imnet_dir  ,
    'num_classes': 1000,
    'wrapper': torchvision.datasets.ImageNet,
    'batch_size': 100,
    'type': 'imnet',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
    "reset_sample":60000,
    "train_sample":60000,
    "train_for_test_sample": 120000,
}

cifar50_224 = {
    'dir': datadir,
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 50,
    'type': 'cifar224',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
    "train_sample":100,
}

cifar5_224 = {
    'dir': datadir,
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 50,
    'type': 'cifar224',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
    "train_sample":1000,
}

cifar50_384 = {
    'dir': datadir,
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 50,
    'type': 'cifar384',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 2,
    "train_sample":100,
}

cifar5_384 = {
    'dir': datadir,
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 50,
    'type': 'cifar384',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
    "train_sample":1000,
}

cifar10 = {
    'dir': datadir,
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
}

imagenet1k = {
    'dir': './data/ffcv/',
    'num_classes': 1000,
    'wrapper': ImageNet1k,
    'batch_size': 16,
    'res': 224,
    'inception_norm': True,
    'shuffle_test': False,
    'type': 'imagenet',
    'num_workers': 8,
}