import os

import argparse

parser = argparse.ArgumentParser('Generate mudsc encoder')

parser.add_argument('--suffix', type=str,
                        help='config name')

parser.add_argument('--gpu', default='0',type=str,
                        help='gpu')

args = parser.parse_args()

print("suffix", args.suffix,"setting gpu",args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from eval_utils import prepare_resetbns_dataloader, load_resetbns_models
from copy import deepcopy
import pickle as pkl
import mudsc
import torch

suffix = args.suffix

result_dir = f"weights/fusion_encoder{suffix}"
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


def generate_task_pairs(tasks):
    task_pair = []
    for i in range(len(tasks) - 1):
        for j in range(i + 1, len(tasks)):
            task_pair.append([tasks[i], tasks[j]])
    return task_pair


batch_size = 100


def get_activate_loader():
    if "act" in suffix:
        resetbns_data_domain = [
            "allensville", "beechwood", "benevolence", "coffeen", "cosmos",
            "forkland", "hanson", "hiteman"
        ]
        return prepare_resetbns_dataloader(domains=resetbns_data_domain,
                                          batch_size=batch_size)
    return None


def add_task_dict(task1, task2):
    if f"{task1}__{task2}" not in result:
        result[f"{task1}__{task2}"] = {}
    if f"{task2}__{task1}" not in result:
        result[f"{task2}__{task1}"] = {}


zipit_dir = "weights/zipit_encoder"
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
    if "zipit" in suffix:
        print("Loading zipit result")
        zipit_path = os.path.join(zipit_dir, f"{task1}__{task2}",
                                  "encoder.pth")
        zipit_dict = torch.load(zipit_path, map_location="cpu")
        task_dicts = [{} for _ in range(len(encoders))]
        for k, v in zipit_dict.items():
            for t, w in zip(task_dicts, v):
                t[k] = w
        for m, t in zip(encoders, task_dicts):
            m.load_state_dict(t)
    for m in models:
        m.eval().cuda()
    perm = mudsc.get_resnet_perm(use_bn_stat="usebn" in suffix,
                                         ingore_in="igin" in suffix,
                                         scaled_bn_mean="sbm" in suffix)
    a = 0.5
    b = 1
    if "lowa" in suffix:
        a = 0.0001
    print("a", a, "b", b)
    fix_rate = 0.5
    if "fr04" in suffix:
        fix_rate = 0.4
    elif "fr03" in suffix:
        fix_rate = 0.3
    avg_state_dict = mudsc.WeightFusion(
        a=a,
        b=b,
        use_cos="cos" in suffix,
        no_fusion="nofu" in suffix,
        fix_sims="fs" in suffix,
        use_permute="useperm" in suffix,
        fix_rate=fix_rate).transform(encoders,
                                     perm,
                                     act_loader=get_activate_loader(),
                                     in_weight_space="iws" in suffix)

    torch.save(avg_state_dict, state_dict_dir)
    print("Save to", state_dict_dir)
