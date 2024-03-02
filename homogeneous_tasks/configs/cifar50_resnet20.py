config = {
    'dataset': {
        'name': 'cifar50',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet20x8',
        'dir': './checkpoints/cifar50_clip/resnet20x8/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}