config = {
    'dataset': {
        'name': 'cifar50',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet20gnx16',
        'dir': './checkpoints/cifar50_clip/resnet20gnx16_wrong/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}