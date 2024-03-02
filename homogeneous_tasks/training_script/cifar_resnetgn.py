import os

import argparse

parser = argparse.ArgumentParser('Training CIFAR')

parser.add_argument('--lr', default=0.1, type=float,
                        help='config name')

parser.add_argument('--gpu', default='0',type=str,
                        help='gpu')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import clip
import torch
from copy import deepcopy

import numpy as np

from utils import *
from sklearn.model_selection import train_test_split
from models.resnets import resnet20gn
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import torch.backends.cudnn as cudnn


def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


reset_random(seed=0)
# INITIALIZATIONS
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
normalize = T.Normalize(np.array(CIFAR_MEAN) / 255, np.array(CIFAR_STD) / 255)
denormalize = T.Normalize(-np.array(CIFAR_MEAN) / np.array(CIFAR_STD),
                          255 / np.array(CIFAR_STD))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_clip = True
    model_dir = f'./checkpoints/cifar50_{"clip" if use_clip else "logits"}'

    split_runs = 5  # num times to sample disjoint split sets
    models_per_run = 1  # num models to train per split
    data_dir = './data'
    wrapper = torchvision.datasets.CIFAR100
    num_classes = 100  # num classes in dataset
    batch_size = 500  # batch size
    model_width = 16  # resnet model width
    epochs = 250  # train epochs

    model_dir = os.path.join(model_dir, f'resnet20gnx{model_width}',
                             'pairsplits')
    print(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(), normalize
    ])
    test_transform = T.Compose([T.ToTensor(), normalize])
    train_dset = wrapper(root=data_dir,
                         train=True,
                         download=True,
                         transform=train_transform)
    test_dset = wrapper(root=data_dir,
                        train=False,
                        download=True,
                        transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)

    if 'clip' in model_dir:
        clip_features = load_clip_features(test_dset.classes, device=device)
        out_dim = 512
    else:
        out_dim = num_classes

    for _ in range(split_runs):
        splits = train_test_split(np.arange(num_classes),
                                  train_size=(num_classes // 2))
        split_trainers = [
            torch.utils.data.DataLoader(torch.utils.data.Subset(
                train_dset, [
                    i for i, label in enumerate(train_dset.targets)
                    if label in split
                ]),
                                        batch_size=500,
                                        shuffle=True,
                                        num_workers=8) for split in splits
        ]

        split_testers = [
            torch.utils.data.DataLoader(torch.utils.data.Subset(
                test_dset, [
                    i for i, label in enumerate(test_dset.targets)
                    if label in split
                ]),
                                        batch_size=500,
                                        shuffle=False,
                                        num_workers=8) for split in splits
        ]
        split1, split2 = splits
        label_remapping = np.zeros(num_classes, dtype=int)
        label_remapping[split1] = np.arange(len(split1))
        label_remapping[split2] = np.arange(len(split2))
        label_remapping = torch.from_numpy(label_remapping)
        print("label remapping: {}".format(label_remapping))
        print(f"{split1}, {split2}")
        for j in range(models_per_run):
            for i in range(len(splits)):
                model = resnet20gn(w=model_width,
                                   num_classes=out_dim).cuda().train()

                if 'clip' in model_dir:
                    class_vectors = [clip_features[split] for split in splits]
                    model, final_acc = train_cliphead(
                        model=model,
                        train_loader=split_trainers[i],
                        test_loader=split_testers[i],
                        class_vectors=class_vectors[i],
                        remap_class_idxs=label_remapping,
                        epochs=epochs,
                        lr=args.lr)
                else:
                    model, final_acc = train_logits(
                        model=model,
                        train_loader=split_trainers[i],
                        test_loader=split_testers[i],
                        epochs=epochs,
                        lr=args.lr)

                print(f'Base model on {splits[i]} Acc: {final_acc}')
                print('Saving Base Model')
                idxs = [str(k) for k in splits[i]]
                save_dir = os.path.join(model_dir, '_'.join(idxs))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir,
                    f'resnet20gnx{model_width}_v{len(os.listdir(save_dir))}.pth.tar'
                )
                save_model(model, save_path)

    print('Done!')