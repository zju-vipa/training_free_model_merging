import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from eval_utils import  load_resetbns_models, prepare_resetbns_dataloader
from visualpriors import visualpriors
import zipit.resnet_graph as graph_module
from zipit.matching_functions import match_tensors_zipit
from zipit.model_merger import ModelMerge, get_merging_fn
from zipit.metric_calculators import get_metric_fns
from copy import deepcopy
import pickle as pkl
import torch
import numpy as np
import random

result_dir = "weights/zipit_encoder"
batch_size = 100
tasks = [
    'class_object',
    'segment_semantic',
    'depth_zbuffer',
    'depth_euclidean',
    'edge_occlusion',
    'edge_texture',
    'keypoints2d',
    'keypoints3d',
    'reshading',
    'normal',
    'autoencoding',
    "denoising",
]
domains = ["ihlen", "mcdade", "muleshoe", "noxapater", "uvalda"]
result = {}

os.makedirs(result_dir, exist_ok=True)


def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_task_pairs(tasks):
    task_pair = []
    for i in range(len(tasks) - 1):
        for j in range(i + 1, len(tasks)):
            task_pair.append([tasks[i], tasks[j]])
    return task_pair


def add_task_dict(task1, task2):
    if f"{task1}__{task2}" not in result:
        result[f"{task1}__{task2}"] = {}
    if f"{task2}__{task1}" not in result:
        result[f"{task2}__{task1}"] = {}


resetbns_data_domain = [
    "allensville", "beechwood", "benevolence", "coffeen", "cosmos", "forkland",
    "hanson", "hiteman"
]
activate_loader = prepare_resetbns_dataloader(domains=resetbns_data_domain,
                                             batch_size=batch_size)

task_pairs = generate_task_pairs(tasks)
for idx, (task1, task2) in enumerate(task_pairs):
    print("Testing task", task1, task2, idx, "/", len(task_pairs))
    save_path = os.path.join(result_dir, f"{task1}__{task2}")
    os.makedirs(save_path, exist_ok=True)
    state_dict_dir = os.path.join(save_path, "encoder.pth")
    if os.path.exists(state_dict_dir):
        print(state_dict_dir, "has exist")
        continue
    models = [
        deepcopy(m)
        for m in load_resetbns_models([task1, task2])
    ]
    encoders = [m.encoder for m in models]
    sample_model = deepcopy(encoders[0])
    Grapher = graph_module.taskonomy_encoder
    graphs = [
        Grapher(deepcopy(base_model)).graphify() for base_model in encoders
    ]

    for m in models:
        m.eval().cuda()
    Merge = ModelMerge(*graphs, device="cuda")
    reset_random()
    other_params = {"a": 0.5, "b": 1}
    avg_state_dict = Merge.transform(sample_model,
                                     activate_loader,
                                     transform_fn=match_tensors_zipit,
                                     metric_classes=get_metric_fns(
                                         ['covariance', 'mean']),
                                     stop_at=None,
                                     **other_params)

    torch.save(avg_state_dict, state_dict_dir)
    print("Save to", state_dict_dir)
