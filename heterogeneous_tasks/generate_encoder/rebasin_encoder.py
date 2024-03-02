import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from eval_utils import evaluate_model, reset_bn_stats, prepare_resetbns_dataloader
from visualpriors import visualpriors
from copy import deepcopy
import rebasin
import torch

result_dir = "weights/rebasin_encoder"
batch_size = 100
tasks = [
    'class_object', 'segment_semantic', 'depth_zbuffer', 'depth_euclidean',
    'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d',
    'curvature', 'reshading', 'normal', 'autoencoding', "denoising"
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


def add_task_dict(task1, task2):
    if f"{task1}__{task2}" not in result:
        result[f"{task1}__{task2}"] = {}
    if f"{task2}__{task1}" not in result:
        result[f"{task2}__{task1}"] = {}


task_pairs = generate_task_pairs(tasks)
for idx, (task1, task2) in enumerate(task_pairs):
    print("Testing task", task1, task2, idx, "/", len(task_pairs))
    save_path = os.path.join(result_dir, f"{task1}__{task2}")
    os.makedirs(save_path, exist_ok=True)
    state_dict_dir = os.path.join(save_path, "encoder.pth")
    if os.path.exists(state_dict_dir):
        print(state_dict_dir, "has exist")
        continue
    models = [deepcopy(m) for m in visualpriors.load_models([task1, task2])]
    encoders = [m.encoder for m in models]
    # for m in models:
    #     m.eval().cuda()
    merge_model = deepcopy(encoders[0])
    perm = rebasin.get_resnet_perm()
    perm_mats, axis2perm = rebasin.weight_match(encoders[0], encoders[1], perm)
    params_a_adapt, params_b_match, params_mask = rebasin.apply_perm(
        encoders[0].state_dict(), encoders[1].state_dict(), perm_mats,
        axis2perm)
    rebasin.network_adapt(merge_model, perm_mats, axis2perm)
    # avg_state_dict = rebasin.avg_param_partial(params_a_adapt, params_b_match,
    #                                            params_mask)

    torch.save(
        {
            task1: params_a_adapt,
            task2: params_b_match,
            "mask": params_mask
        }, state_dict_dir)
    print("Save to", state_dict_dir)
