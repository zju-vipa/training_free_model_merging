config = {
    'dataset': {
        'name': 'cifar50_224',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet50gntimm',
        'dir': './checkpoints/cifar50_clip/resnet50gntimm/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}