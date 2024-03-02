config = {
    'dataset': {
        'name': 'imnet',
        'shuffle_train': True
    },
    'model': {
        'name': 'resnet50dino',
        'dir': './checkpoints/imnet_clip/resnet50dino/pairsplits',
        'bases': []
    },
    'eval_type': 'clip',
}