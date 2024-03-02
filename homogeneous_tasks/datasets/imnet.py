import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import pickle as pkl
import pdb

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

################################################# Global Variables #################################################

####################################################################################################################
from torch.utils.data import DataLoader, Subset
import torch


def prepare_train_loaders(config):

    # return None
    with open("task_split.pkl", "rb") as f:
        task_split_dict = pkl.load(f)

    img_size = 224

    if 'no_transform' in config:
        train_transform = T.Compose([
            T.Resize(int(img_size * 1.143)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dset_raw = config['wrapper'](root=config['dir'],
                                       split="train",
                                       transform=train_transform)
    
    split_pair = config['class_splits']
    print("split pair", split_pair)
    all_split = task_split_dict[split_pair[0]] + task_split_dict[split_pair[1]]
    train_dset = Subset(train_dset_raw,[
        i for i, label in tqdm(enumerate(train_dset_raw.targets))
        if label in all_split
    ])
    
    generator = torch.Generator()
    generator.manual_seed(0)
    indices = torch.randperm(len(train_dset),
                             generator=generator)
    indices_sample = indices[:config["train_sample"]]
    indices_reset = indices[:config["reset_sample"]]
    loaders = {
        'full':
        DataLoader(train_dset,
                   batch_size=config['batch_size'],
                   shuffle=config['shuffle_train'],
                   num_workers=config['num_workers']),
        'sample':
        DataLoader(Subset(train_dset, indices_sample),
                   batch_size=config['batch_size'],
                   shuffle=config['shuffle_train'],
                   num_workers=config['num_workers']),
        'reset':
        DataLoader(Subset(train_dset, indices_reset),
                   batch_size=config['batch_size'],
                   shuffle=config['shuffle_train'],
                   num_workers=config['num_workers'])
    }

    if 'class_splits' in config:
        loaders['splits'] = []
        grouped_class_indices = np.zeros(config['num_classes'], dtype=int)
        for i, splits in enumerate(config['class_splits']):
            splits_ = task_split_dict[splits]
            valid_examples = [
                i for i, label in tqdm(enumerate(train_dset_raw.targets))
                if label in splits_
            ]
            data_subset = Subset(train_dset_raw, valid_examples)
            loaders['splits'].append(
                DataLoader(data_subset,
                           batch_size=config['batch_size'],
                           shuffle=config['shuffle_train'],
                           num_workers=config['num_workers']))
            
            grouped_class_indices[splits_] = np.arange(len(splits_))

        loaders["label_remapping"] = torch.from_numpy(grouped_class_indices)
        loaders['class_splits'] = [task_split_dict[v] for v in config['class_splits']]

    return loaders


def prepare_test_loaders(config):
    
    with open("task_split.pkl", "rb") as f:
        task_split_dict = pkl.load(f)
    
    img_size = 224
    test_transform = T.Compose([
        T.Resize(int(img_size * 1.143)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dset_raw = config['wrapper'](root=config['dir'],
                                      split="val",
                                      transform=test_transform)
    train_dset_raw = config['wrapper'](root=config['dir'],
                                       split="train",
                                       transform=test_transform)

    split_pair = config['class_splits']
    all_split = task_split_dict[split_pair[0]] + task_split_dict[split_pair[1]]
    test_dset = Subset(test_dset_raw, [
        i for i, label in tqdm(enumerate(test_dset_raw.targets))
        if label in all_split
    ])
    train_dset = Subset(train_dset_raw,[
        i for i, label in tqdm(enumerate(train_dset_raw.targets))
        if label in all_split
    ])
    if "train_for_test_sample" in config:
        print(f"sample train data for test", config["train_for_test_sample"])
        generator = torch.Generator()
        generator.manual_seed(0)
        indices = torch.randperm(len(train_dset),
                                generator=generator)
        indices_sample = indices[:config["train_for_test_sample"]]
        train_dset = Subset(train_dset, indices_sample)

    loaders = {
        'full':
        DataLoader(test_dset,
                   batch_size=config['batch_size'],
                   shuffle=config["shuffle_test"],
                   num_workers=config['num_workers']),
        'train':
        DataLoader(train_dset,
                   batch_size=config['batch_size'],
                   shuffle=config["shuffle_test"],
                   num_workers=config['num_workers']),
    }

    if 'class_splits' in config:
        loaders['splits'] = []
        for i, splits in enumerate(config['class_splits']):
            splits = task_split_dict[splits]
            valid_examples = [
                i for i, label in tqdm(enumerate(test_dset_raw.targets))
                if label in splits
            ]
            data_subset = Subset(test_dset_raw, valid_examples)
            loaders['splits'].append(
                DataLoader(data_subset,
                           batch_size=config['batch_size'],
                           shuffle=False,
                           num_workers=config['num_workers']))

    loaders['class_names'] = [x[0] for x in test_dset_raw.classes]
    return loaders
