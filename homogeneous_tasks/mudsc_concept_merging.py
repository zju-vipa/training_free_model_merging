import os

import argparse

parser = argparse.ArgumentParser('Evaluation mudsc merge')

parser.add_argument('--config-name', type=str,
                        help='config name')


parser.add_argument('--suffix', type=str,
                        help='config name')

parser.add_argument('--gpu', default='0',type=str,
                        help='gpu')

args = parser.parse_args()

print("config name", args.config_name,"suffix",args.suffix,"setting gpu",args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import random
from time import time
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import gc
from utils import inject_pair,prepare_experiment_config,reset_bn_stats,write_to_csv
from utils import evaluate_model,CONCEPT_TASKS,flatten_nested_dict,get_config_from_name,find_runable_pairs
import mudsc as weight_fusion
import torch.backends.cudnn as cudnn
import pickle as pkl


suffix = args.suffix


def reset_random(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import pandas as pd


def dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, list):
            if k not in d:
                d[k] = []
            d[k] += v
        else:
            d[k] = v
    return d


def create_df(search_results):
    base = {}
    for _, results in search_results.items():
        base = dict_update(base, results)

    numbers = np.array(list(base.values())).T
    cols = list(base.keys())

    df = pd.DataFrame(numbers, columns=cols)
    return df


def get_task_mapping(labels, splits):
    task_mapping = []
    for i, label in enumerate(labels):
        for j, split in enumerate(splits):
            if label in split:
                task_mapping.append(j)
    return torch.from_numpy(np.array(task_mapping))


def run_node_experiment(node_config, experiment_config, pairs, csv_file):
    for i, pair in enumerate(tqdm(pairs, desc='Evaluating Pairs...')):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(raw_config)
        train_loader = config['data']['train']['full']
        reset_random()
        base_models = [
            reset_bn_stats(base_model, train_loader)
            for base_model in config['models']['bases']
        ]
        print("size", len(base_models))
        config['node'] = node_config
        if "resnet20gn" in config_name:
            vit_perm = weight_fusion.get_resnet_perm_group()
        elif "resnet26" in config_name:
            vit_perm = weight_fusion.get_resnet_perm(block=3,
                                                 num_blocks=[2, 2, 2, 2],
                                                 shortcut_name="downsample",
                                                 fc_name="fc",
                                                 pool_name="global_pool",
                                                 res_start_layer=0)
        elif "resnet50dino" in config_name:
            vit_perm = weight_fusion.get_resnet_perm(
                block=3,
                num_blocks=[3, 4, 6, 3],
                shortcut_name="downsample",
                fc_name="fc",
                res_start_layer=0)
        elif "resnet50gn" in config_name:
            vit_perm = weight_fusion.get_resnet_perm_group(
                block=3,
                num_blocks=[3, 4, 6, 3],
                shortcut_name="downsample",
                fc_name="fc",
                pool_name="global_pool",
                group_num=32,
                res_start_layer=0)
        elif "vit" in config_name:
            vit_perm = weight_fusion.get_vit_perm()
        elif "resnet20" in config_name:
            vit_perm = weight_fusion.get_resnet_perm()
        else:
            raise NotImplementedError

        wf = weight_fusion.WeightFusion(
            a=node_config["params"]["a"],
            b=node_config["params"]["b"],
            fix_rate=node_config["params"]["fix_rate"],
            fix_sims="fs" in suffix,
            no_fusion="avg" in suffix,
            use_permute="useperm" in suffix,
        )
        reset_random()

        def get_activate_loader():
            if "act" in suffix:
                return config['data']['train']["sample"] if "sample" in config[
                    'data']['train'] else train_loader
            return None

        merge_model = wf.transform(base_models,
                                   vit_perm,
                                   act_loader=get_activate_loader(),
                                   in_weight_space="iws" in suffix)
        reset_random()
        reset_bn_stats(merge_model, train_loader)
        results = evaluate_model(experiment_config['eval_type'],
                                 merge_model,
                                 config,
                                 test_train="train" in suffix)
        for idx, split in enumerate(pair):
            results[f'Split {CONCEPT_TASKS[idx]}'] = split
        # results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = "partial_fusion"
        results['Model Name'] = config['model']['name']
        results.update(flatten_nested_dict(node_config, sep=' '))
        write_to_csv(results, csv_file=csv_file)
        print(results)
        # pdb.set_trace()

    print(f'Results of {node_config}: {results}')
    return results


if __name__ == "__main__":
    partial_layer = "all"
    stop_node_map = {7: 21, 13: 42, 19: 63, "all": None}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_name = args.config_name
    skip_pair_idxs = []

    experiment_configs = []
    if "fs" in suffix:
        # for fr in [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]: # search more balanced factor
        # for fr in [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91]: # search more balanced factor
        for fr in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:

            experiment_configs.append({
                'stop_node': stop_node_map[partial_layer],
                'params': {
                    'a': 0.5,
                    'b': 1,
                    "fix_rate": fr
                }
            })
    else:
        experiment_configs.append({
            'stop_node': stop_node_map[partial_layer],
            'params': {
                'a': 0.5,
                'b': 1,
                "fix_rate": 0.5
            }
        })

    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    run_pairs = find_runable_pairs(model_dir,
                                   model_name,
                                   skip_pair_idxs=skip_pair_idxs)
    print(run_pairs)
    csv_file = os.path.join(
        './csvs', raw_config['dataset']['name'], raw_config['model']['name'],
        raw_config['eval_type'],
        f'mudsc_{config_name}_{partial_layer}_configurations{suffix}.csv')

    with torch.no_grad():
        for node_config in experiment_configs:
            raw_config['dataset'].update(node_config.get('dataset', {}))
            run_node_experiment(node_config=node_config,
                                experiment_config=raw_config,
                                pairs=run_pairs,
                                csv_file=csv_file)
            gc.collect()
