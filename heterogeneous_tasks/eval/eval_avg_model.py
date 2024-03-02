from eval_utils import evaluate_model, reset_bn_stats, prepare_resetbns_dataloader
from visualpriors import visualpriors
from copy import deepcopy
import pickle as pkl
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
result_dir = "results/avg"
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
for idx, (task1, task2) in enumerate(task_pairs):
    print("Eval Avg Model Testing task", task1, task2, idx, "/",
          len(task_pairs))
    models = None

    def prepares_models():
        global models
        models = [
            deepcopy(m) for m in visualpriors.load_models([task1, task2])
        ]

        encoders = [m.encoder.state_dict() for m in models]
        avg_state_dict = deepcopy(encoders[0])
        keys = avg_state_dict.keys()
        for st in encoders[1:]:
            for k in keys:
                avg_state_dict[k] = avg_state_dict[k] + st[k]
        for k in keys:
            avg_state_dict[k] = avg_state_dict[k] / len(models)
        del avg_state_dict['compress1.weight']
        for m in models:
            m.encoder.load_state_dict(avg_state_dict, strict=False)
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
