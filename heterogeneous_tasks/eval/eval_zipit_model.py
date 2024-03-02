import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from eval_utils import evaluate_model, reset_bn_stats, prepare_resetbns_dataloader
from visualpriors import visualpriors
from copy import deepcopy
import pickle as pkl
import torch

suffix = ""

result_dir = f"results/zipit{suffix}"
batch_size = 100
tasks = [
    'class_object', 'segment_semantic', 'depth_zbuffer', 'depth_euclidean',
    'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d',
     'reshading', 'normal', 'autoencoding', "denoising"
]
domains = ["ihlen", "mcdade", "muleshoe", "noxapater", "uvalda"]
result = {}

os.makedirs(result_dir, exist_ok=True)
result_file_dir = os.path.join(result_dir, "loss.pkl")

if os.path.exists(result_file_dir):
    print("Loading exist")
    with open(result_file_dir, "rb") as f:
        result = pkl.load(f)
resetbns_data_domain = [
    "allensville", "beechwood", "benevolence", "coffeen", "cosmos", "forkland",
    "hanson", "hiteman"
]
resetbns_loader = prepare_resetbns_dataloader(domains=resetbns_data_domain,
                                          batch_size=batch_size)


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
encoder_weight_root = f"weights/zipit_encoder{suffix}"
for idx, (task1, task2) in enumerate(task_pairs):
    print("Testing task", task1, task2, idx, "/", len(task_pairs))
    models = None

    def prepares_models():
        global models
        if models is not None:
            return
        models = [
            deepcopy(m) for m in visualpriors.load_models([task1, task2])
        ]

        state_dict_dir = os.path.join(encoder_weight_root, f"{task1}__{task2}",
                                      "encoder.pth")
        fusion_state_dict = torch.load(state_dict_dir, map_location="cpu")
        task1_dict = {}
        task2_dict = {}
        for wk, ws in fusion_state_dict.items():
            if "compress" not in wk:
                w_final = sum(w for w in ws) / len(ws)
                task1_dict[wk] = w_final
                task2_dict[wk] = w_final
            else:
                task1_dict[wk] = ws[0]
                task2_dict[wk] = ws[1]
        merged_state_dicts = [task1_dict, task2_dict]
        for m, d in zip(models, merged_state_dicts):
            m.encoder.load_state_dict(d)
            m.eval().cuda()
        reset_bn_stats(models, resetbns_loader)

    add_task_dict(task1, task2)
    for d in domains:
        if d in result[f"{task1}__{task2}"] and d in result[
                f"{task2}__{task1}"]:
            t1 = result[f"{task1}__{task2}"][d]
            t2 = result[f"{task2}__{task1}"][d]
            print(f"Find {d} in {task1}__{task2} {t1}")
            print(f"Find {d} in {task2}__{task1} {t2}")
            continue
        prepares_models()
        res = evaluate_model(models=models,
                             batch_size=batch_size,
                             tasks=[task1, task2],
                             domains=[d])
        result[f"{task1}__{task2}"][d] = res[task2]
        result[f"{task2}__{task1}"][d] = res[task1]
        print(f"Loss {task1} --> {task2}", res[task2])
        print(f"Loss {task2} --> {task1}", res[task1])

        with open(result_file_dir, "wb") as f:
            pkl.dump(result, f)
