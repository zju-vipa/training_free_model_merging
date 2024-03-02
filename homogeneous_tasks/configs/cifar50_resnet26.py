config = {
    'dataset': {
        'name': 'cifar50_224',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet26timm',
        'dir': './checkpoints/cifar50_clip/resnet26timm/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}