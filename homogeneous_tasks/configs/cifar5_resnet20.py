config = {
    'dataset': {
        'name': 'cifar5',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet20x8',
        'dir': './checkpoints/cifar5_clip/resnet20x8/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}