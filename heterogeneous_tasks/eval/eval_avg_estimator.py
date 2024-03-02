import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from eval_utils import evaluate_model, generate_avg_estimator, AvgEstimator
from copy import deepcopy
import pickle as pkl

result_dir = "results/avg_estimator"
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
domains = ["muleshoe", "ihlen", "mcdade", "noxapater", "uvalda"]
result = {}
resetbns_data_domain = [
    "allensville", "beechwood", "benevolence", "coffeen", "cosmos", "forkland",
    "hanson", "hiteman"
]

os.makedirs(result_dir, exist_ok=True)
result_file_dir = os.path.join(result_dir, "loss.pkl")

if os.path.exists(result_file_dir):
    print("Loading exist result")
    with open(result_file_dir, "rb") as f:
        result = pkl.load(f)

weights_path = "weights/avg_estimator"
generate_avg_estimator(tasks=tasks,
                       domains=domains,
                       save_root=weights_path,
                       batch_size=batch_size,
                       num_workers=8)


def add_task_dict(task):
    if task not in result:
        result[task] = {}


for d in domains:
    models = [AvgEstimator(d, t, avg_cache=weights_path) for t in tasks]
    res = evaluate_model(models=models,
                         batch_size=batch_size,
                         tasks=tasks,
                         domains=[d])
    for idx, task in enumerate(tasks):
        add_task_dict(task)
        result[task][d] = res[task]
        print(f"Loss({d}) {task}", res[task])

    with open(result_file_dir, "wb") as f:
        pkl.dump(result, f)
