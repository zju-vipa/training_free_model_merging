config = {
    'dataset': {
        'name': 'cifar50_384',
        'shuffle_train': True
    },
    'model': {
        'name': 'vits',
        'dir': './checkpoints/cifar50_clip/vits/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}