import torch
from torch import nn
from copy import deepcopy, copy
import pdb
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import optimize
import numpy as np
import timm.models.vision_transformer

IGNORE_CFG = {"ignore": None}
IN_CFG = {"in": None}

LINEAR_IN_CFG = {"in": {"dim": -1}}


def gen_reshape_norm_qk():

    def reshape_norm_qk(x):
        x = x.permute(0, 1, 3, 2)
        B, H, D, _ = x.shape
        return x.reshape(B, H * D, -1).contiguous()

    return reshape_norm_qk


def gen_reshape_group_conv(group):

    def reshape_group_conv(x):
        B, C, H, W = x.shape
        Og = C // group
        x = x.reshape(B, group, C // group,
                      -1).permute(0, 2, 1, 3).reshape(B, Og, -1).contiguous()
        return x

    return reshape_group_conv


RESHAPE_FUNC = {
    "norm_qk": gen_reshape_norm_qk,
    "group_conv": gen_reshape_group_conv
}


def group_cfg(group, same_members=False):
    return {"group": {"num": group, "same": same_members}}


def split_dim_cfg(axis, params_name, suffixes):
    """
    order 0 q 1 k 2 v
    """
    return {"split_dim": [axis, params_name, suffixes]}


def act_module_cfg(name=None, dim=None, hook=None, reshape_func=None):
    cfg = {}
    if name is not None:
        cfg["name"] = name
    if dim is not None:
        cfg["dim"] = dim
    if hook is not None:
        cfg["hook"] = hook
    if reshape_func is not None:
        cfg["reshape_func"] = reshape_func
    return {"act_module": cfg}


def get_resnet_perm_group(block=2,
                          num_blocks=[3, 3, 3],
                          shortcut_name="shortcut",
                          fc_name="linear",
                          pool_name='avgpool',
                          res_start_layer=1,
                          group_num=8):
    perm = {}
    cur_perm = 0
    perm[cur_perm] = [
        [0, "conv1.weight"],
        [0, "bn1.weight", group_cfg(group_num)],
        [0, "bn1.bias"],
    ]

    res_perm = cur_perm
    cur_perm += 1
    for l, block_num in enumerate(num_blocks):
        layer_id = l + 1

        for b in range(block_num):
            for c in range(1, block + 1):
                if l >= res_start_layer and b == 0 and c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                        [
                            1, f"layer{layer_id}.{b}.{shortcut_name}.0.weight",
                            IN_CFG
                        ],
                    ])

                elif c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                else:
                    perm[cur_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                    cur_perm += 1
                t = [
                    [0, f"layer{layer_id}.{b}.conv{c}.weight"],
                    [
                        0, f"layer{layer_id}.{b}.bn{c}.weight",
                        group_cfg(group_num)
                    ],
                    [0, f"layer{layer_id}.{b}.bn{c}.bias"],
                ]
                if c < block:
                    perm[cur_perm] = t
                else:
                    if b == 0 and l >= res_start_layer:
                        res_perm = cur_perm
                        cur_perm += 1
                        perm[res_perm] = [
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.0.weight"
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.weight",
                                group_cfg(group_num)
                            ],
                            [0, f"layer{layer_id}.{b}.{shortcut_name}.1.bias"],
                        ]
                    perm[res_perm].extend(t)

    perm[res_perm].extend([
        [1, f"{fc_name}.weight", {
            "in": pool_name
        }],
    ])
    return perm


def get_resnet_perm(block=2,
                    num_blocks=[3, 3, 3],
                    shortcut_name="shortcut",
                    fc_name="linear",
                    pool_name='avgpool',
                    res_start_layer=1,
                    ignore_running_val=True):
    running_cfg = IGNORE_CFG if ignore_running_val else {}
    perm = {}
    cur_perm = 0
    perm[cur_perm] = [
        [0, "conv1.weight"],
        [0, "bn1.weight"],
        [0, "bn1.bias"],
        [0, "bn1.running_mean", running_cfg],
        [0, "bn1.running_var", running_cfg],
    ]

    res_perm = cur_perm
    cur_perm += 1
    for l, block_num in enumerate(num_blocks):
        layer_id = l + 1

        for b in range(block_num):
            for c in range(1, block + 1):
                if l >= res_start_layer and b == 0 and c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                        [
                            1, f"layer{layer_id}.{b}.{shortcut_name}.0.weight",
                            IN_CFG
                        ],
                    ])

                elif c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                else:
                    perm[cur_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                    cur_perm += 1
                t = [
                    [0, f"layer{layer_id}.{b}.conv{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.bias"],
                    [
                        0, f"layer{layer_id}.{b}.bn{c}.running_mean",
                        running_cfg
                    ],
                    [0, f"layer{layer_id}.{b}.bn{c}.running_var", running_cfg],
                ]
                if c < block:
                    perm[cur_perm] = t
                else:
                    if b == 0 and l >= res_start_layer:
                        res_perm = cur_perm
                        cur_perm += 1
                        perm[res_perm] = [
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.0.weight"
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.weight"
                            ],
                            [0, f"layer{layer_id}.{b}.{shortcut_name}.1.bias"],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_mean",
                                running_cfg
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_var",
                                running_cfg
                            ],
                        ]
                    perm[res_perm].extend(t)

    perm[res_perm].extend([
        [1, f"{fc_name}.weight", {
            "in": pool_name
        }],
    ])
    return perm


def get_resnext_perm(num_blocks=[3, 3, 3],
                     shortcut_name="shortcut",
                     fc_name="linear",
                     pool_name='avgpool',
                     res_start_layer=0,
                     group=8,
                     ignore_running_val=True):
    block = 3
    running_cfg = IGNORE_CFG if ignore_running_val else {}
    perm = {}
    cur_perm = 0
    perm[cur_perm] = [
        [0, "conv1.weight"],
        [0, "bn1.weight"],
        [0, "bn1.bias"],
        [0, "bn1.running_mean", running_cfg],
        [0, "bn1.running_var", running_cfg],
    ]

    res_perm = cur_perm
    cur_perm += 1
    for l, block_num in enumerate(num_blocks):
        layer_id = l + 1

        for b in range(block_num):
            for c in range(1, block + 1):
                if l >= res_start_layer and b == 0 and c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                        [
                            1, f"layer{layer_id}.{b}.{shortcut_name}.0.weight",
                            IN_CFG
                        ],
                    ])

                elif c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", IN_CFG],
                    ])
                else:

                    in_cfg = {
                        **IN_CFG,
                        **act_module_cfg(reshape_func=("group_conv", {
                            "group": group
                        }))
                    } if c == 2 else IN_CFG
                    perm[cur_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", in_cfg],
                    ])
                    cur_perm += 1
                if c == 1:
                    t = []
                    for weight_name in [
                            f"layer{layer_id}.{b}.conv{c}.weight",
                            f"layer{layer_id}.{b}.bn{c}.weight",
                            f"layer{layer_id}.{b}.bn{c}.bias"
                    ]:
                        for g in range(group):
                            cfg = [0, f"{weight_name}_{g}"]
                            if g == 0:
                                cfg.append(
                                    split_dim_cfg(0, weight_name,
                                                  list(range(group))))
                            t.append(cfg)
                    for weight_name in [
                            f"layer{layer_id}.{b}.bn{c}.running_mean",
                            f"layer{layer_id}.{b}.bn{c}.running_var",
                    ]:
                        for g in range(group):
                            cfg = [0, f"{weight_name}_{g}", running_cfg]
                            if g == 0:
                                cfg[2] = {
                                    **cfg[2],
                                    **split_dim_cfg(0, weight_name,
                                                    list(range(group)))
                                }
                            t.append(cfg)
                else:
                    t = [
                        [0, f"layer{layer_id}.{b}.conv{c}.weight"],
                        [0, f"layer{layer_id}.{b}.bn{c}.weight"],
                        [0, f"layer{layer_id}.{b}.bn{c}.bias"],
                        [
                            0, f"layer{layer_id}.{b}.bn{c}.running_mean",
                            running_cfg
                        ],
                        [
                            0, f"layer{layer_id}.{b}.bn{c}.running_var",
                            running_cfg
                        ],
                    ]
                if c < block:
                    perm[cur_perm] = t
                else:
                    if b == 0 and l >= res_start_layer:
                        res_perm = cur_perm
                        cur_perm += 1
                        perm[res_perm] = [
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.0.weight"
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.weight"
                            ],
                            [0, f"layer{layer_id}.{b}.{shortcut_name}.1.bias"],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_mean",
                                running_cfg
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_var",
                                running_cfg
                            ],
                        ]
                    perm[res_perm].extend(t)

    perm[res_perm].extend([
        [1, f"{fc_name}.weight", {
            "in": pool_name
        }],
    ])
    return perm


def get_vit_perm(num_layer=12, num_heads=6,ignore_head=False):
    vit_perm = {}
    cur_perm = 0
    vit_perm[cur_perm] = [
        [2, "cls_token"],
        [2, "pos_embed"],
        [0, "patch_embed.proj.weight"],
        [0, "patch_embed.proj.bias"],
    ]
    hold_perm = cur_perm
    cur_perm += 1
    for i in range(num_layer):
        vit_perm[hold_perm].extend([
            [0, f"blocks.{i}.norm1.weight"],
            [0, f"blocks.{i}.norm1.bias"],
            [1, f"blocks.{i}.attn.qkv.weight_q", LINEAR_IN_CFG],
            [1, f"blocks.{i}.attn.qkv.weight_k", LINEAR_IN_CFG],
            [1, f"blocks.{i}.attn.qkv.weight_v", LINEAR_IN_CFG],
            # first res
            [0, f"blocks.{i}.attn.proj.weight"],
            [0, f"blocks.{i}.attn.proj.bias"],
            [0, f"blocks.{i}.norm2.weight"],
            [0, f"blocks.{i}.norm2.bias"],
            #second res
            [1, f"blocks.{i}.mlp.fc1.weight", LINEAR_IN_CFG],
            [0, f"blocks.{i}.mlp.fc2.weight"],
            [0, f"blocks.{i}.mlp.fc2.bias"],
        ])
        vit_perm[cur_perm] = [
            [
                0, f"blocks.{i}.attn.qkv.weight_q", {
                    **group_cfg(num_heads),
                    **split_dim_cfg(0, f"blocks.{i}.attn.qkv.weight", [
                        "q", "k", "v"
                    ]),
                    **act_module_cfg(f"blocks.{i}.attn.k_norm",
                                     reshape_func=("norm_qk", {}),
                                     hook="post")
                }
            ],
            [
                0, f"blocks.{i}.attn.qkv.bias_q",
                split_dim_cfg(0, f"blocks.{i}.attn.qkv.bias", ["q", "k", "v"])
            ],
            [
                0, f"blocks.{i}.attn.qkv.weight_k",
                act_module_cfg(f"blocks.{i}.attn.k_norm",
                               reshape_func=("norm_qk", {}),
                               hook="post")
            ],
            [0, f"blocks.{i}.attn.qkv.bias_k"],
        ]
        cur_perm += 1
        vit_perm[cur_perm] = [
            [0, f"blocks.{i}.attn.qkv.weight_v",
             group_cfg(num_heads)],
            [0, f"blocks.{i}.attn.qkv.bias_v"],
            [1, f"blocks.{i}.attn.proj.weight", LINEAR_IN_CFG],
        ]

        cur_perm += 1
        vit_perm[cur_perm] = [[0, f"blocks.{i}.mlp.fc1.weight"],
                              [0, f"blocks.{i}.mlp.fc1.bias"],
                              [1, f"blocks.{i}.mlp.fc2.weight", LINEAR_IN_CFG]]
        cur_perm += 1
    head_cfg = IN_CFG
    if ignore_head:
        head_cfg = {**head_cfg,**IGNORE_CFG}
    vit_perm[hold_perm].extend([
        [0, "norm.weight"],
        [0, "norm.bias"],
        [1, "head.weight", head_cfg],
    ])
    return vit_perm


def get_vit_clip_perm(num_layer=12, num_heads=12,ignore_head=False):
    vit_perm = {}
    cur_perm = 0
    vit_perm[cur_perm] = [
        [2, "cls_token"],
        [2, "pos_embed"],
        [0, "patch_embed.proj.weight"],
        [0, "norm_pre.weight"],
        [0, "norm_pre.bias"],
    ]
    hold_perm = cur_perm
    cur_perm += 1
    for i in range(num_layer):
        vit_perm[hold_perm].extend([
            [0, f"blocks.{i}.norm1.weight"],
            [0, f"blocks.{i}.norm1.bias"],
            [1, f"blocks.{i}.attn.qkv.weight_q", LINEAR_IN_CFG],
            [1, f"blocks.{i}.attn.qkv.weight_k", LINEAR_IN_CFG],
            [1, f"blocks.{i}.attn.qkv.weight_v", LINEAR_IN_CFG],
            # first res
            [0, f"blocks.{i}.attn.proj.weight"],
            [0, f"blocks.{i}.attn.proj.bias"],
            [0, f"blocks.{i}.norm2.weight"],
            [0, f"blocks.{i}.norm2.bias"],
            #second res
            [1, f"blocks.{i}.mlp.fc1.weight", LINEAR_IN_CFG],
            [0, f"blocks.{i}.mlp.fc2.weight"],
            [0, f"blocks.{i}.mlp.fc2.bias"],
        ])
        vit_perm[cur_perm] = [
            [
                0, f"blocks.{i}.attn.qkv.weight_q", {
                    **group_cfg(num_heads),
                    **split_dim_cfg(0, f"blocks.{i}.attn.qkv.weight", [
                        "q", "k", "v"
                    ]),
                    **act_module_cfg(f"blocks.{i}.attn.k_norm",
                                     reshape_func=("norm_qk", {}),
                                     hook="post")
                }
            ],
            [
                0, f"blocks.{i}.attn.qkv.bias_q",
                split_dim_cfg(0, f"blocks.{i}.attn.qkv.bias", ["q", "k", "v"])
            ],
            [
                0, f"blocks.{i}.attn.qkv.weight_k",
                act_module_cfg(f"blocks.{i}.attn.k_norm",
                               reshape_func=("norm_qk", {}),
                               hook="post")
            ],
            [0, f"blocks.{i}.attn.qkv.bias_k"],
        ]
        cur_perm += 1
        vit_perm[cur_perm] = [
            [0, f"blocks.{i}.attn.qkv.weight_v",
             group_cfg(num_heads)],
            [0, f"blocks.{i}.attn.qkv.bias_v"],
            [1, f"blocks.{i}.attn.proj.weight", LINEAR_IN_CFG],
        ]

        cur_perm += 1
        vit_perm[cur_perm] = [[0, f"blocks.{i}.mlp.fc1.weight"],
                              [0, f"blocks.{i}.mlp.fc1.bias"],
                              [1, f"blocks.{i}.mlp.fc2.weight", LINEAR_IN_CFG]]
        cur_perm += 1
    head_cfg = IN_CFG
    if ignore_head:
        head_cfg = {**head_cfg,**IGNORE_CFG}
    vit_perm[hold_perm].extend([
        [0, "norm.weight"],
        [0, "norm.bias"],
        [1, "head.weight", head_cfg],
    ])
    return vit_perm

def refine_perm(perm):
    weight_cfg = {}
    old_perm = perm
    perm = {}
    in_perm = {}
    for p, o in old_perm.items():
        if isinstance(o, list):
            o = {"weights": o}
        # 补全配置信息
        for wl in o["weights"]:
            if len(wl) < 3:
                wl.append({})
            axis, w, cfg = wl
            if w not in weight_cfg:
                weight_cfg[w] = []
            weight_cfg[w].append([axis, p, cfg])
            if "in" in cfg:
                if w not in in_perm:
                    in_perm[w] = []
                in_perm[w].append(p)
        # 添加激活节点
        weights_num = len(o["weights"])
        weights = o["weights"]
        for axis, w, cfg in weights:
            if "act_module" in cfg:
                if "act_modules" not in o:
                    o["act_modules"] = []
                cfg_act = deepcopy(cfg["act_module"])
                if "name" not in cfg_act:
                    cfg_act["name"] = ".".join(w.split(".")[:-1])
                if "dim" not in cfg_act:
                    cfg_act["dim"] = 1
                if "hook" not in cfg_act:
                    cfg_act["hook"] = "pre"
                if "reshape_func" not in cfg_act:
                    cfg_act["reshape_func"] = None
                o["act_modules"].append(cfg_act)

        if "act_modules" not in o:
            # 以每个节点最后一个输入模块作为统计激活值的模块
            for i in range(weights_num - 1, -1, -1):
                axis, w, cfg = weights[i]
                if "in" in cfg:
                    cfg_in = cfg["in"]
                    if cfg_in is None:
                        cfg_in = {}
                    else:
                        cfg_in = deepcopy(cfg_in)
                    if isinstance(cfg["in"], str):
                        cfg_in = dict(name=cfg_in, dim=1)
                    if "name" not in cfg_in:
                        cfg_in["name"] = ".".join(w.split(".")[:-1])
                    if "dim" not in cfg_in:
                        cfg_in["dim"] = 1
                    if "hook" not in cfg_in:
                        cfg_in["hook"] = "pre"
                    if "reshape_func" not in cfg_in:
                        cfg_in["reshape_func"] = None
                    o["act_modules"] = [cfg_in]
                    break
        # 添加分组标记
        for axis, w, cfg in weights:
            if "group" in cfg:
                if "group" not in o:
                    o["group"] = cfg["group"]
                else:
                    o_cfg = o["group"]
                    cur_cfg = cfg["group"]
                    assert all(
                        [o_cfg[k] == v for k, v in cur_cfg.items()]
                    ), f"Group config in the same layer should be the same.{o_cfg} v.s. {cur_cfg}"
        # 添加split dim的标记
        for axis, w, cfg in weights:
            if "split_dim" in cfg:
                if "split_dim" not in o:
                    o["split_dim"] = []
                o["split_dim"].append(cfg["split_dim"])

        perm[p] = o
    for p, o in perm.items():
        if "in" not in o:
            o["in"] = []
        for axis, w, cfg in o["weights"]:
            if "in" not in cfg and w in in_perm:
                o["in"].extend(in_perm[w])
        o["in"] = list(set(o["in"]))
    return perm, weight_cfg


def remove_col(x, idx, temp=None):

    if temp is None:
        return torch.cat([x[:, :idx], x[:, idx + 1:]], dim=-1)
    else:
        R, C = x.shape
        temp = temp[:R, :C]
        _, L = x[:, idx + 1:].shape
        temp[:, :L] = x[:, idx + 1:]
        x[:, idx:idx + L] = temp[:, :L]
        return x[:, :C - 1]


class CovarianceMetric:

    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
        self.numel = 0

    def update(self, *feats, **aux_params):
        batch_size, feature_dim = feats[0].shape[0], feats[0].shape[1]
        self.numel += batch_size
        feats = [
            torch.transpose(f, 0, 1).reshape(feature_dim, -1) for f in feats
        ]
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)

        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]

        if self.mean is None: self.mean = torch.zeros_like(mean)
        if self.outer is None: self.outer = torch.zeros_like(outer)
        if self.std is None: self.std = torch.zeros_like(std)

        self.mean += mean * batch_size
        self.outer += outer * batch_size
        self.std += std * batch_size

    def finalize(self, eps=1e-7):
        self.outer /= self.numel
        self.mean /= self.numel
        self.std /= self.numel
        cov = self.outer - torch.outer(self.mean, self.mean)
        if torch.isnan(cov).any():
            breakpoint()
        if (torch.diagonal(cov) < 0).sum():
            pdb.set_trace()
        std = torch.diagonal(cov).sqrt()
        cov = cov / (torch.clamp(torch.outer(std, std), min=eps))
        return cov


class WeightFusion(object):

    def __init__(self,
                 reduce=0.5,
                 a=0.5,
                 b=1,
                 relax=5,
                 iter=100,
                 use_cos=False,
                 no_fusion=False,
                 fix_sims=False,
                 fix_rate=0.5,
                 use_permute=False,
                 verbose=True) -> None:
        self.reduce = reduce
        self.a = a
        self.b = b
        self.relax = relax
        self.iter = iter
        self.fix_sims = fix_sims
        self.sims_dict = None
        self.fix_rate = fix_rate
        self.verbose = verbose
        # self.scaled_fix = scaled_fix
        if fix_sims:
            print(f"Fix Similarity {fix_rate}")
        self.compute_correlation = self._cossim if use_cos else self._correlation
        if use_cos:
            print("use cosine similarity")
        self.fusion_weight = self.default_fusion_weight
        self.group_fusion_weight = self.default_group_fusion_weight
        if use_permute:
            self.fusion_weight = self.permute_weight
            self.group_fusion_weight = self.group_permute_weight
            print("permute weight")

        if no_fusion:
            self.fusion_weight = self.no_fusion_weight
            self.group_fusion_weight = self.no_fusion_weight
            print("nofusion")

    def transform(self,
                  nets,
                  perm,
                  act_loader=None,
                  in_weight_space=False,
                  return_state_dict=False,
                  random_state=0):

        self.fusion_model: nn.Module = deepcopy(nets[0])
        perm = deepcopy(perm)
        self.perm, self.weight_cfg = refine_perm(perm)
        self.perm_names = list(perm.keys())
        self.perm_mats = {}
        self.params = [{k: v
                        for k, v in net.state_dict().items()} for net in nets]
        self.split_dim(self.perm, self.params)  # split qkv for fusion
        if act_loader is not None:
            self.act_transform(nets, act_loader)
        in_weight_space = in_weight_space or act_loader is None
        if in_weight_space:
            self.iter_transform(random_state=random_state)
        merged_state_dict = self.get_merged_state_dict()
        self.merge_dim(self.perm,
                       [merged_state_dict])  # split qkv for inference
        self.network_adapt_by_state_dict(merged_state_dict, self.fusion_model)
        self.fusion_model.load_state_dict(merged_state_dict)
        if return_state_dict:
            return self.get_merged_state_dict(no_avg=True)
        return self.fusion_model

    def gen_act_sims(self, nets, act_loader: DataLoader):
        sims_dict = {k: CovarianceMetric() for k in self.perm}
        device = list(nets[0].parameters())[0].device
        hooks = []
        feats = [{} for _ in range(len(nets))]

        def add_hooks(net: nn.Module, idx):
            modules = {k: v for k, v in net.named_modules()}

            def prehook_gen(perm_name,
                            act_dim,
                            m_idx,
                            reshape_func,
                            act_module_name=None):
                if perm_name not in feats[idx]:
                    feats[idx][perm_name] = {}
                if reshape_func is not None:
                    reshape_func = RESHAPE_FUNC[reshape_func[0]](
                        **reshape_func[1])

                def prehook(m, x):
                    x = x[0].detach()
                    # print("pre", act_module_name, x.shape)
                    if reshape_func is None:
                        if act_dim != 1:
                            x = torch.moveaxis(x, act_dim, 1)
                    else:
                        x = reshape_func(x)
                    feats[idx][perm_name][m_idx] = x
                    return None

                return prehook

            def posthook_gen(perm_name, act_dim, m_idx, reshape_func):
                if perm_name not in feats[idx]:
                    feats[idx][perm_name] = {}
                if reshape_func is not None:
                    reshape_func = RESHAPE_FUNC[reshape_func[0]](
                        **reshape_func[1])

                def posthook(m, x_in, x):
                    # print("post",x.shape,x_in.shape)
                    x = x.detach()
                    if reshape_func is None:
                        if act_dim != 1:
                            x = torch.moveaxis(x, act_dim, 1)
                    else:
                        x = reshape_func(x)
                    # print("post post",x.shape)
                    feats[idx][perm_name][m_idx] = x
                    return None

                return posthook

            for k, o in self.perm.items():
                for m_idx, act_module in enumerate(o["act_modules"]):
                    act_module_name, act_dim, hook_type, reshape_func = act_module[
                        "name"], act_module["dim"], act_module[
                            "hook"], act_module["reshape_func"]
                    # print("regist", act_module_name)
                    module = modules[act_module_name]
                    if hook_type == "pre":
                        hooks.append(
                            module.register_forward_pre_hook(
                                prehook_gen(k,
                                            act_dim,
                                            m_idx,
                                            reshape_func,
                                            act_module_name=act_module_name)))
                    elif hook_type == "post":
                        hooks.append(
                            module.register_forward_hook(
                                posthook_gen(k, act_dim, m_idx, reshape_func)))
                    else:
                        raise NotImplementedError

        def clear_hooks():
            for h in hooks:
                h.remove()

        for i, net in enumerate(nets):
            net.eval().cuda()
            add_hooks(net, i)
        with torch.no_grad():
            for img, _ in tqdm(act_loader, desc="Computing activation"):
                img = img.cuda()
                for net in nets:
                    net(img)
                for k, s in sims_dict.items():
                    fs = [
                        torch.stack([v.float() for v in f[k].values()], dim=-1)
                        for f in feats
                    ]
                    s.update(*fs)
        clear_hooks()
        for net in nets:
            net.to(device)

        return {k: v.finalize() for k, v in sims_dict.items()}

    def act_transform(self, nets, act_loader: DataLoader):
        sims_dict = self.gen_act_sims(nets, act_loader)
        if self.fix_sims:
            self.sims_dict = sims_dict
        for k, sims in tqdm(sims_dict.items(),
                            desc="Computing permutation",
                            total=len(sims_dict)):
            perm_cfg = self.perm[k]
            group = 1
            same_members = None
            if "group" in perm_cfg:
                group_cfg = perm_cfg["group"]
                group = group_cfg["num"]
                same_members = group_cfg["same"]
            if group == 1:
                merge, unmerge, merge_value = self.fusion_weight(
                    sims,
                    r=self.reduce,
                    a=self.a,
                    b=self.b,
                    get_merge_value=True)
            else:
                merge, unmerge, merge_value = self.group_fusion_weight(
                    sims,
                    r=self.reduce,
                    a=self.a,
                    b=self.b,
                    group=group,
                    get_merge_value=True)
            merge = merge * len(self.params)
            self.perm_mats[k] = (merge, unmerge)

    def iter_transform(self, random_state=0, tol=5):

        no_progress_count = 0
        generator = torch.Generator()
        generator.manual_seed(random_state)
        perm_state = {p: 0 for p in self.perm_names}
        total_oldL = 0
        self.best_perm_mats = None
        for iter in range(self.iter):
            progress = False

            for p_ix in torch.randperm(len(self.perm_names),
                                       generator=generator):
                p = self.perm_names[p_ix]
                perm_cfg = self.perm[p]
                wvs = self.get_weight_vectors(p)
                sims = self.compute_correlation(wvs)
                if self.fix_sims and self.sims_dict is not None:
                    sims_act = self.sims_dict[p]
                    # if self.scaled_fix:
                    #     sims = sims / sims.abs().max()
                    #     sims_act = sims_act / sims_act.abs().max()

                    sims = sims * self.fix_rate + sims_act * (1 -
                                                              self.fix_rate)
                    # sims = self.sims_dict[p]
                group = 1
                same_members = None
                if "group" in perm_cfg:
                    group_cfg = perm_cfg["group"]
                    group = group_cfg["num"]
                    same_members = group_cfg["same"]
                if group == 1:
                    merge, unmerge, merge_value = self.fusion_weight(
                        sims,
                        r=self.reduce,
                        a=self.a,
                        b=self.b,
                        get_merge_value=True)
                else:
                    merge, unmerge, merge_value = self.group_fusion_weight(
                        sims,
                        r=self.reduce,
                        a=self.a,
                        b=self.b,
                        group=group,
                        get_merge_value=True)
                merge = merge * len(self.params)
                # assert torch.diagonal(merge).all(), "no full"

                if merge_value > perm_state[p]:
                    progress = True
                perm_state[p] = merge_value

                self.perm_mats[p] = (merge, unmerge)
            total_newL = sum(v for v in perm_state.values()) / len(perm_state)
            if self.verbose:
                print("iter", iter, "process count", no_progress_count, "newL",
                      total_newL)
            if total_oldL >= total_newL and self.best_perm_mats is not None:
                no_progress_count += 1
                if no_progress_count >= tol:
                    break
                self.perm_mats = copy(self.best_perm_mats)
            else:
                no_progress_count = 0
                self.best_perm_mats = copy(self.perm_mats)
                total_oldL = total_newL

        self.perm_mats = self.best_perm_mats
        self.best_perm_mats = None

    def perm_(self, axis, ws, perm_mat, cfg=None):
        ws_perm = []
        perm_mat = perm_mat.chunk(len(self.params), dim=0)
        for i, w in enumerate(ws):
            # print("weight", w.shape, "mat", perm_mat[i].shape)

            w = torch.transpose(w, axis, 0)
            shape = [x for x in w.shape]
            merge_mat = perm_mat[i].T
            raw_dim = shape[0]
            new_dim = merge_mat.shape[0]
            shape[0] = new_dim
            w = torch.matmul(merge_mat, w.reshape(raw_dim, -1)).reshape(*shape)
            ws_perm.append(torch.transpose(w, 0, axis))
        return ws_perm

    @staticmethod
    def split_dim(fined_perm: dict, params):
        """
        Break params in certain dimension. For example, break qkv into q and k and v
        """
        for o in fined_perm.values():
            if "split_dim" not in o:
                continue
            params_need_split = o["split_dim"]
            for axis, p_name, suffixes in params_need_split:
                for p in params:
                    w = p[p_name]
                    feature_dim = w.shape[axis] // len(suffixes)
                    assert feature_dim * len(suffixes) == w.shape[
                        axis], "Feature number to splited should be divisible by len(suffixes)"
                    ws = torch.chunk(w, len(suffixes), dim=axis)
                    for w, w_suffix in zip(ws, suffixes):
                        p[f"{p_name}_{w_suffix}"] = w

    @staticmethod
    def merge_dim(fined_perm: dict, params: dict):
        """
        Merge params which are broken in split_dim()
        """
        for o in fined_perm.values():
            if "split_dim" not in o:
                continue
            params_need_split = o["split_dim"]
            for axis, p_name, suffixes in params_need_split:
                p_name_suffixes = [f"{p_name}_{s}" for s in suffixes]
                for p in params:
                    p[p_name] = torch.concat([p[ps] for ps in p_name_suffixes],
                                             dim=axis)
                    for ps in p_name_suffixes:
                        del p[ps]

    def get_weight_vectors(self, p):
        o = self.perm[p]
        weight_vectors = [[] for _ in range(len(self.params))]
        for axis, w, cfg in o["weights"]:
            if "ignore" in cfg:
                continue
            ws = [param[w] for param in self.params]
            if "scaled_bn" in cfg:
                w_var = w.split(".")
                w_var[-1] = "running_var"
                w_var = ".".join(w_var)
                w_vars = [
                    torch.sqrt((param[w_var] + 1e-7)) for param in self.params
                ]
                ws = [w1 / w2 for w1, w2 in zip(ws, w_vars)]
            ignore_weight = False
            for a_, p_, cfg_ in self.weight_cfg[w]:
                # if axis == a_:
                #     continue
                # if p_ not in self.perm_mats:
                #     ignore_weight = True
                #     break
                if axis == a_:
                    continue
                elif p_ in self.perm_mats:
                    select = 0 if "in" in cfg_ else 1
                    perm_mat = self.perm_mats[p_][select]
                    ws = self.perm_(a_, ws, perm_mat, cfg_)
            if ignore_weight:
                continue
            for i, w_ in enumerate(ws):
                n = w_.shape[axis]
                weight_vectors[i].append(
                    torch.transpose(w_, axis, 0).reshape(n, -1))
        weight_vectors = [torch.concat(wv, dim=1) for wv in weight_vectors]
        return weight_vectors

    def _correlation(self, feats):
        feats = torch.cat(feats, dim=0)
        # print("feats cat",feats.shape)
        feats = torch.nan_to_num(feats, 0, 0, 0)

        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]

        cov = outer - torch.outer(mean, mean)
        if torch.isnan(cov).any():
            breakpoint()
        if (torch.diagonal(cov) < 0).sum():
            pdb.set_trace()

        std = torch.diagonal(cov).sqrt()
        cov = cov / (torch.clamp(torch.outer(std, std), min=1e-7))
        # print("covariacne done", covariance.shape) # covariacne done torch.Size([1024, 1024])
        return cov

    def _cossim(self, feats):
        feats = torch.cat(feats, dim=0)
        # print("feats cat",feats.shape)
        feats_norm = torch.norm(feats, dim=1, keepdim=True)
        feats = feats / feats_norm.clamp_min(1e-8)
        outer = (feats @ feats.T)

        # print("covariacne done", covariance.shape) # covariacne done torch.Size([1024, 1024])
        return outer

    def permute_weight(self, sims, r=.5, get_merge_value=False, **kwargs):
        """
        Matches arbitrary models by permuting all to the space of the first in your graph list. 
        Mimics Rebasin methods. 
        Hyperparameters and return are as defined in match_tensors_zipit.
        """
        correlation = sims
        O = correlation.shape[0]

        N = len(self.params)
        Om = O // N
        device = correlation.device

        mats = [torch.eye(Om, device=device)]
        merge_value = []
        for i in range(1, N):
            try:
                row_ind, col_ind = optimize.linear_sum_assignment(
                    correlation[:Om, Om * i:Om * (i + 1)].cpu().numpy(),
                    maximize=True)
            except:
                pdb.set_trace()
            mats.append(
                torch.eye(
                    Om,
                    device=device)[torch.tensor(col_ind).long().to(device)].T)
            merge_value.append(
                correlation[:Om, Om * i:Om *
                            (i + 1)].cpu().numpy()[row_ind,
                                                   col_ind].mean().item())

        unmerge_mats = mats

        unmerge = torch.cat(unmerge_mats, dim=0)
        merge = torch.cat(mats, dim=0)
        merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    def group_permute_weight(self,
                             sims,
                             r=.5,
                             group=1,
                             get_merge_value=False,
                             same_members=False,
                             **kwargs):
        """
        Matches arbitrary models by permuting all to the space of the first in your graph list. 
        Mimics Rebasin methods. 
        Hyperparameters and return are as defined in match_tensors_zipit.
        """
        correlation: torch.Tensor = sims
        O = correlation.shape[0]

        N = len(self.params)
        Om = O // N
        Og = Om // group
        # print("Dim", O, N, Om, Og)
        assert Og * group == Om, f"Number of features {Om} shoule be divisible by group number {group}"
        device = correlation.device

        mats = [torch.eye(Om, device=device)]
        merge_value = []
        for i in range(1, N):
            group_corr = np.empty((group, group), dtype=np.float32)
            group_cols = np.empty((group, group, Og), dtype=np.int64)
            try:
                sub_correlation = correlation[:Om,
                                              Om * i:Om * (i + 1)].reshape(
                                                  group, Og, group,
                                                  Og).permute(0, 2, 1,
                                                              3).cpu().numpy()
                for j in range(group):
                    for k in range(group):
                        group_corr_mat = sub_correlation[j, k]
                        row_ind, col_ind = optimize.linear_sum_assignment(
                            group_corr_mat, maximize=True)
                        group_corr[j, k] = group_corr_mat[row_ind,
                                                          col_ind].mean()
                        group_cols[j, k, :] = col_ind + k * Og
                row_ind, col_ind = optimize.linear_sum_assignment(
                    group_corr, maximize=True)
                merge_value.append(group_corr[row_ind, col_ind].mean().item())
                col_ind = group_cols[row_ind, col_ind, :].reshape(-1)
            except:
                pdb.set_trace()
            mats.append(
                torch.eye(
                    Om,
                    device=device)[torch.tensor(col_ind).long().to(device)].T)

        unmerge_mats = mats

        unmerge = torch.cat(unmerge_mats, dim=0)
        merge = torch.cat(mats, dim=0)
        merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    def default_fusion_weight(
        self,
        sims,
        r=.5,
        a=0,
        b=1,
        get_merge_value=False,
    ):
        sims = torch.clone(sims)
        O = sims.shape[0]
        remainder = int(O * (1 - r) + 1e-4)

        permutation_matrix = torch.eye(O, O)  #, device=sims.device)

        torch.diagonal(sims)[:] = -torch.inf  # 禁止自身匹配
        num_models = len(self.params)
        Om = O // num_models
        original_model = torch.zeros(O, device=sims.device).long()
        for i in range(num_models):
            original_model[i * Om:(i + 1) * Om] = i

        to_remove = permutation_matrix.shape[1] - remainder
        budget = torch.zeros(num_models, device=sims.device).long() + int(
            (to_remove // num_models) * b + 1e-4)
        # if True:
        #     # print("ban same model match")
        #     for i in range(num_models):
        #         sims[Om * i:Om * (i + 1), Om * i:Om * (i + 1)] = -torch.inf
        merge_value = []

        while permutation_matrix.shape[1] > remainder:
            best_idx = sims.reshape(-1).argmax().item()
            row_idx = best_idx % sims.shape[1]
            col_idx = best_idx // sims.shape[1]
            merge_value.append(sims[row_idx, col_idx].item())

            if col_idx < row_idx:
                row_idx, col_idx = col_idx, row_idx

            row_origin = original_model[row_idx]  # row属于第几个模型
            col_origin = original_model[col_idx]  # col属于第几个模型

            permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
            permutation_matrix = remove_col(permutation_matrix, col_idx)

            sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:,
                                                                    col_idx])

            if a <= 0:
                # 匹配过行列不在发生匹配
                sims[row_origin * Om:(row_origin + 1) * Om,
                     row_idx] = -torch.inf
                sims[col_origin * Om:(col_origin + 1) * Om,
                     row_idx] = -torch.inf
            else:
                sims[:, row_idx] *= a  # 匹配过的元素减少被匹配的可能
            sims = remove_col(
                sims, col_idx)  # 表示col_idx 和 row_idx被合并到一起，此时row_idx就代表col_idx

            sims[row_idx, :] = torch.minimum(sims[row_idx, :],
                                             sims[col_idx, :])
            if a <= 0:
                sims[row_idx,
                     row_origin * Om:(row_origin + 1) * Om] = -torch.inf
                sims[row_idx,
                     col_origin * Om:(col_origin + 1) * Om] = -torch.inf
            else:
                sims[row_idx, :] *= a
            sims = remove_col(sims.T, col_idx).T

            row_origin, col_origin = original_model[row_idx], original_model[
                col_idx]
            original_model = remove_col(original_model[None, :], col_idx)[0]

            if row_origin == col_origin:
                origin = original_model[row_idx].item()
                budget[origin] -= 1

                if budget[origin] <= 0:
                    # kill origin 原模型的预算没了就不在匹配
                    selector = original_model == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf
        unmerge = permutation_matrix
        merge = permutation_matrix / (
            permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    def group_matcher(self, sims: torch.Tensor):
        """
        group_sims: [group, group, Og, Og]
        """
        group = sims.shape[0]
        group_sims = np.empty((group, group))
        for i in range(group):
            for j in range(group):
                group_corr_mat = sims[i, j].detach().cpu().numpy()
                row_ind, col_ind = optimize.linear_sum_assignment(
                    group_corr_mat, maximize=True)
                group_sims[i, j] = group_corr_mat[row_ind,
                                                  col_ind].mean().item()
        row_ind, col_ind = optimize.linear_sum_assignment(group_sims,
                                                          maximize=True)
        drop_group_mask = torch.ones(
            group_sims.shape,
            device=sims.device,
            dtype=torch.bool,
        )
        drop_group_mask[row_ind, col_ind] = False
        return drop_group_mask

    def default_group_fusion_weight(
        self,
        sims,
        r=.5,
        a=0,
        b=1,
        b_group=1,
        get_merge_value=False,
        group=1,
        same_members=False,
    ):
        sims = torch.clone(sims)
        O = sims.shape[0]

        permutation_matrix = torch.eye(O, O)  #, device=sims.device)

        torch.diagonal(sims)[:] = -torch.inf  # prohibit self-matching
        num_models = len(self.params)
        Om = O // num_models
        Og = Om // group
        Ng = num_models * group
        assert Og * group == Om, f"Number of feature {Om} should be divisible by group."

        original_model = torch.zeros(O, device=sims.device, dtype=torch.long)
        for i in range(num_models):
            original_model[i * Om:(i + 1) * Om] = i
        original_group = torch.zeros(O, device=sims.device, dtype=torch.long)
        for i in range(Ng):
            original_group[i * Og:(i + 1) * Og] = i

        group_remainder = int((Og * num_models) * (1 - r) + 1e-4)
        remainder = group * group_remainder
        to_remove = permutation_matrix.shape[1] - remainder
        budget_value = int((to_remove // num_models) * b + 1e-4)
        budget = torch.full((num_models, ),
                            budget_value,
                            device=sims.device,
                            dtype=torch.long)
        group_budget_value = int((to_remove * b_group // Ng) + 1e-4)
        group_budget = torch.full((Ng, ),
                                  group_budget_value,
                                  device=sims.device,
                                  dtype=torch.long).long()
        sims_grouped = sims.reshape(
            num_models, group, Og, num_models, group,
            Og).permute(0, 3, 1, 4, 2,
                        5).cpu().detach().numpy()  # n,n,g,g,Og,Og
        merged_group = [[i] for i in range(group)]
        for i in range(1, num_models):
            sub_sims = sims_grouped[0, i]
            group_corr = np.empty((group, group), dtype=np.float32)
            for j in range(group):
                for k in range(group):
                    sub_sub_sims = sub_sims[j, k]
                    row_idx, col_idx = optimize.linear_sum_assignment(
                        sub_sub_sims, maximize=True)
                    group_corr[j, k] = sub_sub_sims[row_idx, col_idx].mean()
            row_idx, col_idx = optimize.linear_sum_assignment(group_corr,
                                                              maximize=True)
            for r, c in zip(row_idx, col_idx):
                merged_group[r.item()].append(c.item() + i * group)
        merge_value = []

        group_to_remove = 0
        current_group_mask = None
        temp_ = torch.empty_like(permutation_matrix).to(permutation_matrix)

        next_pair = 0
        # print(merged_group)
        while permutation_matrix.shape[1] > remainder:
            if group_to_remove <= 0:

                current_group_mask = torch.full_like(sims,
                                                     -torch.inf,
                                                     dtype=sims.dtype,
                                                     device=sims.device)

                target_group_mask = None
                for current_group in merged_group[next_pair]:
                    if target_group_mask is None:
                        target_group_mask = (original_group == current_group)
                    else:
                        target_group_mask = target_group_mask | (
                            original_group == current_group)
                next_pair += 1
                current_group_mask[target_group_mask[None, :]
                                   & target_group_mask[:, None]] = 0
                group_to_remove = Og * num_models - group_remainder
                # print("group to removed",group_to_remove,group_remainder,"O",O,remainder)

            best_idx = (sims + current_group_mask).reshape(-1).argmax().item()

            row_idx = best_idx % sims.shape[1]
            col_idx = best_idx // sims.shape[1]

            merge_value.append(sims[row_idx, col_idx].item())

            if col_idx < row_idx:
                row_idx, col_idx = col_idx, row_idx

            row_origin = original_model[row_idx]  # which model the row belong to
            col_origin = original_model[col_idx]  # which model the col belong to

            permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
            permutation_matrix = remove_col(permutation_matrix,
                                            col_idx,
                                            temp=temp_)

            sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:,
                                                                    col_idx])

            if a <= 0:
                # Matched rows and columns are no longer matched.
                sims[row_origin * Om:(row_origin + 1) * Om,
                     row_idx] = -torch.inf
                sims[col_origin * Om:(col_origin + 1) * Om,
                     row_idx] = -torch.inf
            else:
                sims[:, row_idx] *= a  # Matched elements reduce the possibility of being matched.
            # It means that col_idx and row_idx are merged, and row_idx stands for col_idx.
            sims = remove_col(sims, col_idx, temp=temp_)
            current_group_mask = remove_col(current_group_mask,
                                            col_idx,
                                            temp=temp_)

            sims[row_idx, :] = torch.minimum(sims[row_idx, :],
                                             sims[col_idx, :])
            if a <= 0:
                sims[row_idx,
                     row_origin * Om:(row_origin + 1) * Om] = -torch.inf
                sims[row_idx,
                     col_origin * Om:(col_origin + 1) * Om] = -torch.inf
            else:
                sims[row_idx, :] *= a
            sims = remove_col(sims.T, col_idx, temp=temp_).T
            current_group_mask = remove_col(current_group_mask.T,
                                            col_idx,
                                            temp=temp_).T
            #
            row_origin, col_origin = original_model[row_idx], original_model[
                col_idx]
            original_model = remove_col(original_model[None, :], col_idx)[0]
            # deduct model budget
            if row_origin == col_origin:
                origin = original_model[row_idx].item()
                budget[origin] -= 1

                if budget[origin] <= 0:
                    # kill origin if the budget of the original model is gone, it will not be matched.
                    selector = original_model == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf
                    # print("model budget out", origin)
            # # deduct group budget
            row_origin, col_origin = original_group[row_idx], original_group[
                col_idx]
            original_group = remove_col(original_group[None, :], col_idx)[0]

            if row_origin == col_origin:
                origin = original_group[row_idx].item()
                group_budget[origin] -= 1

                if group_budget[origin] <= 0:
                    # kill origin if the budget of the original model is gone, it will not be matched.
                    selector = original_group == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf
                    # print("group budget out", origin)
            # The current grouping is merged once.
            group_to_remove -= 1
        permuted_permutation_matrix = []
        for v in merged_group:
            for g in v:
                permuted_permutation_matrix.append(
                    permutation_matrix[:, original_group == g])
        permutation_matrix = torch.concat(permuted_permutation_matrix, dim=1)
        unmerge = permutation_matrix
        merge = permutation_matrix / (
            permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        # a, b = torch.unique(original_group, return_counts=True)
        # cnt = {
        #     k.detach().cpu().item(): v.detach().cpu().item()
        #     for k, v in zip(a, b)
        # }
        # print("Cnt", cnt)
        # print("Merged Group", merged_group)
        # print("Merged Group SUM", {
        #     k: sum(cnt[i] for i in v)
        #     for k, v in enumerate(merged_group)
        # })
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    def default_group_fusion_weight_no(
        self,
        sims,
        r=.5,
        a=0,
        b=1,
        b_group=1,
        get_merge_value=False,
        group=1,
    ):
        sims = torch.clone(sims)
        O = sims.shape[0]

        permutation_matrix = torch.eye(O, O)  #, device=sims.device)

        torch.diagonal(sims)[:] = -torch.inf  
        num_models = len(self.params)
        Om = O // num_models
        Og = Om // group
        Ng = num_models * group
        assert Og * group == Om, f"Number of feature {Om} should be divisible by group."

        original_model = torch.zeros(O, device=sims.device, dtype=torch.long)
        for i in range(num_models):
            original_model[i * Om:(i + 1) * Om] = i
        original_group = torch.zeros(O, device=sims.device, dtype=torch.long)
        for i in range(Ng):
            original_group[i * Og:(i + 1) * Og] = i

        group_remainder = int((Og * num_models) * (1 - r) + 1e-4)
        remainder = group * group_remainder
        to_remove = permutation_matrix.shape[1] - remainder
        budget_value = int((to_remove // num_models) * b + 1e-4)
        budget = torch.full((num_models, ),
                            budget_value,
                            device=sims.device,
                            dtype=torch.long)
        group_budget_value = int((to_remove * b_group // Ng) + 1e-4)
        group_budget = torch.full((Ng, ),
                                  group_budget_value,
                                  device=sims.device,
                                  dtype=torch.long).long()
        merge_value = []
        group_to_remove = 0
        current_group_mask = None
        current_group_row = None
        current_group_col = None
        temp_ = torch.empty_like(permutation_matrix).to(permutation_matrix)
        # temp_ = None
        # print("start group budget", group_budget_value, "model budget",
        #       budget_value)
        merged_group = []
        while permutation_matrix.shape[1] > remainder:
            if group_to_remove <= 0:
                if current_group_mask is not None:
                    # print("mask group", current_group_mask.shape, sims.shape)
                    current_group_mask = (original_group == current_group_row
                                          ) | (original_group
                                               == current_group_col)
                    current_group_mask = current_group_mask[
                        None, :] | current_group_mask[:, None]
                    sims[current_group_mask] = -torch.inf
                id_group_mask = original_group[:, None] == original_group[
                    None, :]  # Exclude the same-group combination

                id_group_offset = torch.zeros_like(sims,
                                                   dtype=sims.dtype,
                                                   device=sims.device)
                id_group_offset[id_group_mask] = -torch.inf
                max_pos = (sims + id_group_offset).reshape(-1).argmax().item()
                current_group_row = max_pos % sims.shape[1]
                current_group_col = max_pos // sims.shape[1]
                if current_group_col < current_group_row:
                    current_group_row, current_group_col = current_group_col, current_group_row

                current_group_row = original_group[current_group_row].item()
                current_group_col = original_group[current_group_col].item()
                assert current_group_row != current_group_col, "Groups to fusion should be different."
                # print("matching group", current_group_row, current_group_col)
                merged_group.append([current_group_row, current_group_col])
                current_group_mask = torch.full_like(sims,
                                                     -torch.inf,
                                                     dtype=sims.dtype,
                                                     device=sims.device)
                target_group_mask = (original_group == current_group_row) | (
                    original_group == current_group_col)
                current_group_mask[target_group_mask[None, :]
                                   & target_group_mask[:, None]] = 0
                group_to_remove = Og * num_models - group_remainder
                # print("group to removed",group_to_remove,group_remainder,"O",O,remainder)

            best_idx = (sims + current_group_mask).reshape(-1).argmax().item()

            row_idx = best_idx % sims.shape[1]
            col_idx = best_idx // sims.shape[1]

            merge_value.append(sims[row_idx, col_idx].item())

            if col_idx < row_idx:
                row_idx, col_idx = col_idx, row_idx

            row_origin = original_model[row_idx] 
            col_origin = original_model[col_idx] 

            permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
            permutation_matrix = remove_col(permutation_matrix,
                                            col_idx,
                                            temp=temp_)

            sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:,
                                                                    col_idx])

            if a <= 0:
                # Matched rows and columns are no longer matched.
                sims[row_origin * Om:(row_origin + 1) * Om,
                     row_idx] = -torch.inf
                sims[col_origin * Om:(col_origin + 1) * Om,
                     row_idx] = -torch.inf
            else:
                sims[:, row_idx] *= a  # Matched elements reduce the possibility of being matched.
            sims = remove_col(sims, col_idx, temp=temp_)
            current_group_mask = remove_col(current_group_mask,
                                            col_idx,
                                            temp=temp_)

            sims[row_idx, :] = torch.minimum(sims[row_idx, :],
                                             sims[col_idx, :])
            if a <= 0:
                sims[row_idx,
                     row_origin * Om:(row_origin + 1) * Om] = -torch.inf
                sims[row_idx,
                     col_origin * Om:(col_origin + 1) * Om] = -torch.inf
            else:
                sims[row_idx, :] *= a
            sims = remove_col(sims.T, col_idx, temp=temp_).T
            current_group_mask = remove_col(current_group_mask.T,
                                            col_idx,
                                            temp=temp_).T
            #
            row_origin, col_origin = original_model[row_idx], original_model[
                col_idx]
            original_model = remove_col(original_model[None, :], col_idx)[0]
            # deduct model budget
            if row_origin == col_origin:
                origin = original_model[row_idx].item()
                budget[origin] -= 1

                if budget[origin] <= 0:
                    selector = original_model == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf
                    # print("model budget out", origin)
            # # deduct group budget
            row_origin, col_origin = original_group[row_idx], original_group[
                col_idx]
            original_group = remove_col(original_group[None, :], col_idx)[0]

            if row_origin == col_origin:
                origin = original_group[row_idx].item()
                group_budget[origin] -= 1

                if group_budget[origin] <= 0:
                    selector = original_group == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf
                    # print("group budget out", origin)
            # The current grouping is merged once.
            group_to_remove -= 1
        permuted_permutation_matrix = []
        for v in merged_group:
            for g in v:
                permuted_permutation_matrix.append(
                    permutation_matrix[:, original_group == g])
        permutation_matrix = torch.concat(permuted_permutation_matrix, dim=1)
        unmerge = permutation_matrix
        merge = permutation_matrix / (
            permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        a, b = torch.unique(original_group, return_counts=True)
        cnt = {
            k.detach().cpu().item(): v.detach().cpu().item()
            for k, v in zip(a, b)
        }
        print("Merged Group", merged_group)
        print("Merged Group SUM",
              {k: cnt[k] + cnt[v]
               for k, v in merged_group})
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    def default_group_fusion_weight_2(
        self,
        sims,
        r=.5,
        a=0,
        b=1,
        get_merge_value=False,
        group=1,
    ):
        sims = torch.clone(sims)
        O = sims.shape[0]

        permutation_matrix = torch.eye(O, O)  #, device=sims.device)

        torch.diagonal(sims)[:] = -torch.inf  # Prohibit self-matching
        num_models = len(self.params)
        Om = O // num_models
        Og = Om // group

        assert Og * group == Om, f"Number of feature {Om} should be divisible by group."

        group_remainder = int((Og * num_models) * (1 - r) + 1e-4)
        remainder = group * group_remainder
        to_remove = permutation_matrix.shape[1] - remainder
        Ng = num_models * group
        sims = sims.reshape(num_models, group, Og, num_models, group,
                            Og).permute(0, 3, 1, 4, 2, 5)  # n,n,g,g,Og,Og
        non_diag_mask = (1 - torch.eye(group, device=sims.device)).bool()
        for i in range(num_models):
            for j in range(i, num_models):
                if i == j:
                    sims[i, j][non_diag_mask] = -torch.inf
                else:
                    drop_group_mask = self.group_matcher(sims[i, j])
                    sims[i, j][drop_group_mask] = -torch.inf
                    sims[j, i][drop_group_mask.T] = -torch.inf
        merged_group = {k: [] for k in range(Ng)}
        sims_where = (sims != -torch.inf).reshape(
            num_models, num_models, group, group,
            -1).any(dim=4).permute(0, 2, 1, 3).reshape(Ng, Ng)
        for i in range(Ng):
            if i not in merged_group:
                continue
            same_groups = torch.argwhere(sims_where[i]).flatten().tolist()
            merged_group[i].extend(same_groups)
            for s in same_groups:
                if s == i:
                    continue
                del merged_group[s]

        sims = sims.permute(0, 2, 4, 1, 3, 5).reshape(O, O)

        original_group = torch.zeros(O, device=sims.device, dtype=torch.long)
        for i in range(Ng):
            original_group[i * Og:(i + 1) * Og] = i
        group_budget_value = int((to_remove * b // Ng) + 1e-4)
        group_budget = torch.full((Ng, ),
                                  group_budget_value,
                                  device=sims.device,
                                  dtype=torch.long).long()
        merge_value = []
        # print("group budget", group_budget_value)
        temp_ = torch.empty_like(permutation_matrix).to(permutation_matrix)

        while permutation_matrix.shape[1] > remainder:

            best_idx = sims.reshape(-1).argmax().item()

            row_idx = best_idx % sims.shape[1]
            col_idx = best_idx // sims.shape[1]

            merge_value.append(sims[row_idx, col_idx].item())

            if col_idx < row_idx:
                row_idx, col_idx = col_idx, row_idx

            permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
            permutation_matrix = remove_col(permutation_matrix,
                                            col_idx,
                                            temp=temp_)

            sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:,
                                                                    col_idx])

            if a <= 0:
                # Matched rows and columns are no longer matched.
                sims[row_origin * Om:(row_origin + 1) * Om,
                     row_idx] = -torch.inf
                sims[col_origin * Om:(col_origin + 1) * Om,
                     row_idx] = -torch.inf
            else:
                sims[:, row_idx] *= a  # Matched elements reduce the possibility of being matched.
            # It means that col_idx and row_idx are merged, and row_idx stands for col_idx.
            sims = remove_col(sims, col_idx, temp=temp_)

            sims[row_idx, :] = torch.minimum(sims[row_idx, :],
                                             sims[col_idx, :])
            if a <= 0:
                sims[row_idx,
                     row_origin * Om:(row_origin + 1) * Om] = -torch.inf
                sims[row_idx,
                     col_origin * Om:(col_origin + 1) * Om] = -torch.inf
            else:
                sims[row_idx, :] *= a
            sims = remove_col(sims.T, col_idx, temp=temp_).T

            # # deduct group budget
            row_origin, col_origin = original_group[row_idx].item(
            ), original_group[col_idx].item()
            original_group = remove_col(original_group[None, :], col_idx)[0]

            if row_origin == col_origin:
                origin = original_group[row_idx].item()
                group_budget[origin] -= 1

                if group_budget[origin] == 0:
                    selector = original_group == origin
                    sims[selector[:, None] & selector[None, :]] = -torch.inf
                    # print("group budget out", origin)
            # elif row_origin in merged_group:
            #     if col_origin not in merged_group[row_origin]:
            #         merged_group[row_origin].extend(merged_group[col_origin])
            #         merged_group[row_origin].append(col_origin)
            #         del merged_group[col_origin]
        permuted_permutation_matrix = []
        for v in merged_group.values():
            for g in v:
                permuted_permutation_matrix.append(
                    permutation_matrix[:, original_group == g])
        permutation_matrix = torch.concat(permuted_permutation_matrix, dim=1)
        unmerge = permutation_matrix
        merge = permutation_matrix / (
            permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        a, b = torch.unique(original_group, return_counts=True)
        cnt = {
            k.detach().cpu().item(): v.detach().cpu().item()
            for k, v in zip(a, b)
        }
        print("Merged Group", merged_group)
        print("Merged Group SUM",
              {k: sum(cnt[i] for i in v)
               for k, v in merged_group.items()})
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    def no_fusion_weight(
        self,
        sims,
        r=.5,
        a=0,
        b=1,
        get_merge_value=False,
        **kwargs,
    ):
        O = sims.shape[0]
        Om = O // len(self.params)
        merge_value = [1]
        permutation_matrix = torch.concat(
            [torch.eye(Om) for _ in range(len(self.params))], dim=0)
        unmerge = permutation_matrix
        merge = permutation_matrix / (
            permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
        merge = merge.to(sims.device)
        unmerge = unmerge.to(sims.device)
        if get_merge_value:
            merge_value = sum(merge_value) / max(len(merge_value), 1e-7)
            return merge, unmerge, merge_value
        return merge, unmerge

    @torch.no_grad()
    def network_adapt(self, net: nn.Module):
        modules = {k: v for k, v in net.named_modules()}
        for wk, v in self.weight_cfg.items():
            wk_path = wk.split(".")
            modules_name = ".".join(wk_path[:-1])
            params_name = wk_path[-1]
            m = modules[modules_name]
            w: nn.Parameter = getattr(m, params_name)

            for axis, p, cfg in v:
                select = 1 if "in" in cfg else 0
                perm_mat = self.perm_mats[p][select]
                shape = [x for x in w.shape]
                new_w = w
                if shape[axis] != perm_mat.shape[1]:
                    shape[axis] = perm_mat.shape[1]
                    new_w = torch.empty(shape).to(new_w)
                w.set_(new_w)
                # if anno is not None:
                #     if "qkv" in anno:
                #         group = anno[1][0]
                #         target_dim = w.shape[axis]
                #         head_dim = target_dim // (group * 3)
                #         attn_module_name = ".".join(wk_path[:-2])
                #         attn_module = modules[attn_module_name]
                #         attn_module.head_dim = head_dim
                if params_name == "weight":
                    if isinstance(m, nn.Linear):
                        m.out_features, m.in_features = new_w.shape[
                            0], new_w.shape[1]
                    elif isinstance(m, nn.Conv2d):
                        # print("new_w",shape)
                        m.out_channels, m.in_channels = new_w.shape[
                            0], new_w.shape[1]
                    elif isinstance(m, nn.LayerNorm):
                        m.normalized_shape = (new_w.shape[0], )
                    elif isinstance(m, nn.BatchNorm2d):
                        m.num_features = new_w.shape[0]
                    elif isinstance(m, nn.GroupNorm):
                        m.num_channels = new_w.shape[0]
                    else:
                        print(type(m))
                        raise NotImplementedError

    @staticmethod
    @torch.no_grad()
    def network_adapt_by_state_dict(state_dict, net: nn.Module):
        modules = {k: v for k, v in net.named_modules()}
        for wk, v in state_dict.items():
            wk_path = wk.split(".")
            modules_name = ".".join(wk_path[:-1])
            params_name = wk_path[-1]
            m = modules[modules_name]
            w: nn.Parameter = getattr(m, params_name)

            new_w = w
            if new_w.shape != v.shape:
                new_w = torch.empty(v.shape).to(new_w)
            w.set_(new_w)
            # if anno is not None:
            #     if "qkv" in anno:
            #         group = anno[1][0]
            #         target_dim = w.shape[axis]
            #         head_dim = target_dim // (group * 3)
            #         attn_module_name = ".".join(wk_path[:-2])
            #         attn_module = modules[attn_module_name]
            #         attn_module.head_dim = head_dim
            if params_name == "weight":
                if isinstance(m, nn.Linear):
                    m.out_features, m.in_features = new_w.shape[
                        0], new_w.shape[1]
                elif isinstance(m, nn.Conv2d):
                    # print("new_w",shape)
                    m.out_channels, m.in_channels = new_w.shape[
                        0], new_w.shape[1]
                elif isinstance(m, nn.LayerNorm):
                    m.normalized_shape = (new_w.shape[0], )
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = new_w.shape[0]
                elif isinstance(m, nn.GroupNorm):
                    m.num_channels = new_w.shape[0]
                else:
                    print(type(m))
                    raise NotImplementedError

    def get_merged_state_dict(self, no_avg=False):
        merged_dict = {}
        rest_keys = set(self.params[0].keys())
        for wk, v in self.weight_cfg.items():
            rest_keys.remove(wk)
            ws = [param[wk] for param in self.params]
            # print("wk", wk)
            for a, p, cfg in v:
                # print("axis", a, "perm", p)
                select = 1 if "in" in cfg else 0
                perm_mat = self.perm_mats[p][select]
                ws = self.perm_(a, ws, perm_mat, cfg)
            if no_avg:
                w_final = ws
            else:
                w_final = sum(w for w in ws) / len(ws)
            merged_dict[wk] = w_final

        for wk in rest_keys:
            print("no permutation weight", wk)
            ws = [param[wk] for param in self.params]
            merged_dict[wk] = sum(w for w in ws) / len(ws)
        return merged_dict

    def apply_permute(self, params, no_avg=False):
        merged_dict = {}
        rest_keys = set(self.params[0].keys())
        for wk, v in self.weight_cfg.items():
            rest_keys.remove(wk)
            ws = [param[wk] for param in params]
            # print("wk", wk)
            for a, p, cfg in v:
                # print("axis", a, "perm", p)
                select = 1 if "in" in cfg else 0
                perm_mat = self.perm_mats[p][select]
                ws = self.perm_(a, ws, perm_mat, cfg)
            if no_avg:
                w_final = ws
            else:
                w_final = sum(w for w in ws) / len(ws)
            merged_dict[wk] = w_final

        for wk in rest_keys:
            print("no permutation weight", wk)
            ws = [param[wk] for param in self.params]
            merged_dict[wk] = sum(w for w in ws) / len(ws)
        return merged_dict
