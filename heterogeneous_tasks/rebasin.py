import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'VGG19': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def get_vgg_perm(cfg_name="VGG11"):
    layers = vgg_cfg[cfg_name]
    perm = {}
    cur_perm = 0
    layer_idx = 0
    for l in layers:
        if isinstance(l, int):
            if cur_perm in perm:
                perm[cur_perm].extend([
                    [1, f"features.{layer_idx}.weight"],
                ])
                cur_perm += 1
            perm[cur_perm] = [
                [0, f"features.{layer_idx}.weight"],
            ]
            layer_idx += 1
        layer_idx += 1
    perm[cur_perm].extend([
        [1, "classifier.weight"],
    ])
    return perm


def get_resnet_perm(
    block=3,
    num_blocks=[3, 4, 6, 3],
    shortcut_name="downsample",
    fc_name="compress1",
    res_Start_layer=0,
):
    perm = {}
    cur_perm = 0
    perm[cur_perm] = [
        [0, "conv1.weight"],
        [0, "bn1.weight"],
        [0, "bn1.bias"],
        [0, "bn1.running_mean"],
        [0, "bn1.running_var"],
    ]
    res_perm = cur_perm
    cur_perm += 1
    for l, block_num in enumerate(num_blocks):
        layer_id = l + 1

        for b in range(block_num):
            for c in range(1, block + 1):
                if l >= res_Start_layer and b == 0 and c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight"],
                        [1, f"layer{layer_id}.{b}.{shortcut_name}.0.weight"],
                    ])

                elif c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight"],
                    ])
                else:
                    perm[cur_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight"],
                    ])
                    cur_perm += 1
                t = [
                    [0, f"layer{layer_id}.{b}.conv{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.bias"],
                    [0, f"layer{layer_id}.{b}.bn{c}.running_mean"],
                    [0, f"layer{layer_id}.{b}.bn{c}.running_var"],
                ]
                if c < block:
                    perm[cur_perm] = t
                else:
                    if b == 0 and l >= res_Start_layer:
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
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_mean"
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.{shortcut_name}.1.running_var"
                            ],
                        ]
                    perm[res_perm].extend(t)

    perm[res_perm].extend([
        [1, f"{fc_name}.weight"],
    ])
    return perm


def padding_weight(w, axis, anno, K=0):
    if w is None:
        return w
    if K == 0:
        return w
    group = 1
    qkv_factor = 1
    if anno is not None and any(a in anno[0] for a in ["qkv", "group"]):
        group = anno[1][0]
        if "qkv" in anno[0]:
            qkv_factor = 3
    raw_dim = len(w.shape)
    if raw_dim == 1:
        w = w.reshape(w.shape[0], 1)

    w = torch.moveaxis(w, axis, 0)
    target_dim = w.shape[0]
    new_target_dim = None
    rest_shape = [x for x in w.shape[1:]]
    w = w.reshape(target_dim, -1)
    rest_dim = w.shape[1]

    head_dim = target_dim // (group * qkv_factor)
    assert head_dim * group * qkv_factor == target_dim
    w = w.reshape(qkv_factor, group, head_dim, rest_dim)
    w = torch.concat([w, torch.zeros(qkv_factor, group, K, rest_dim).to(w)],
                     dim=2)
    new_target_dim = qkv_factor * group * (head_dim + K)

    w = w.reshape(new_target_dim, *rest_shape)
    w = torch.moveaxis(w, 0, axis)
    if raw_dim == 1:
        w = w.reshape(-1)
    return w


def inference_K(w, axis, anno, indexes):
    if w is None:
        return 0
    qkv_factor = 1
    group = 1
    if anno is not None and any(a in anno[0] for a in ["qkv", "group"]):
        group = anno[1][0]
        if "qkv" in anno[0]:
            qkv_factor = 3

    target_dim = w.shape[axis]

    head_dim = target_dim // (qkv_factor * group)
    assert head_dim * qkv_factor * group == target_dim
    actual_dim = len(indexes[0][0])
    K = actual_dim - head_dim
    assert K >= 0
    return K


def _perm(w: torch.Tensor,
          axis,
          indexes,
          m: torch.Tensor = None,
          anno=None,
          wa=None):
    index = None

    ma = None
    K = inference_K(w, axis, anno, indexes)
    w = padding_weight(w, axis, anno, K)
    ma = torch.ones_like(wa) if wa is not None else None
    wa = padding_weight(wa, axis, anno, K)
    ma = padding_weight(ma, axis, anno, K)
    m = padding_weight(m, axis, anno, K)
    if anno is not None:
        if "qkv" in anno[0]:
            cis_all = []
            start_dim = 0
            index_group = indexes[1][0]
            for _ in range(2):
                cis_ = []
                for ci in indexes[2]:
                    cis_.append(ci + start_dim)
                    start_dim += ci.shape[0]
                cis = [None for _ in range(index_group.shape[0])]
                for i, idx in enumerate(index_group):
                    cis[idx] = cis_[i]
                cis_all.extend(cis)
            cis_ = []
            for ci in indexes[0]:
                cis_.append(ci + start_dim)
                start_dim += ci.shape[0]
            cis = [None for _ in range(index_group.shape[0])]
            if not all(i == idx for i, idx in enumerate(index_group)):
                print("head change")
            for i, idx in enumerate(index_group):
                cis[idx] = cis_[i]
            cis_all.extend(cis)
            index = torch.concat(cis_all, axis=0)
        elif "group" in anno[0]:
            cis_ = []
            start_dim = 0
            index_group = indexes[1][0]
            for ci in indexes[0]:
                cis_.append(ci + start_dim)
                start_dim += ci.shape[0]
            cis = [None for _ in range(index_group.shape[0])]
            for i, idx in enumerate(index_group):
                cis[idx] = cis_[i]
            index = torch.concat(cis, axis=0)
    if index is None:
        index = indexes[0][0]
        if w.shape[axis] == index.shape[0] * 2:
            print(
                f"weight shape {w.shape[axis]}, index shape {index.shape[0]}. Will double index shape"
            )
            index = torch.concat([index, index + index.shape[0]])
    index = index.to(w).long()
    w_ = torch.zeros_like(w).to(w)
    m_ = torch.zeros_like(w).to(w)
    wa_index = torch.arange(index.shape[0]).to(w).long()
    w_.index_copy_(axis, index, w.index_select(axis, wa_index))
    if m is not None:
        m_.index_copy_(axis, index, m.index_select(axis, wa_index))
    else:
        m_.index_copy_(axis, index, ma.index_select(axis, wa_index))
    if ma is not None:
        m_[~ma.bool()] = 0
    return w_, m_, wa


def _perm2(w: torch.Tensor,
           axis,
           indexes,
           m: torch.Tensor = None,
           anno=None,
           wa=None,
           wname=None,
           p=None):
    index = None

    ma = None
    K = inference_K(w, axis, anno, indexes)
    w = padding_weight(w, axis, anno, K)
    ma = torch.ones_like(wa) if wa is not None else None
    wa = padding_weight(wa, axis, anno, K)
    ma = padding_weight(ma, axis, anno, K)
    m = padding_weight(m, axis, anno, K)
    if anno is not None:
        if "qkv" in anno[0]:
            cis_all = []
            start_dim = 0
            index_group = indexes[1][0]
            for _ in range(2):
                cis_ = []
                for ci in indexes[2]:
                    cis_.append(ci + start_dim)
                    start_dim += ci.shape[0]
                cis = [None for _ in range(index_group.shape[0])]
                for i, idx in enumerate(index_group):
                    cis[idx] = cis_[i]
                cis_all.extend(cis)
            cis_ = []
            for ci in indexes[0]:
                cis_.append(ci + start_dim)
                start_dim += ci.shape[0]
            cis = [None for _ in range(index_group.shape[0])]
            if not all(i == idx for i, idx in enumerate(index_group)):
                print("head change")
            for i, idx in enumerate(index_group):
                cis[idx] = cis_[i]
            cis_all.extend(cis)
            index = torch.concat(cis_all, axis=0)
        elif "group" in anno[0]:
            cis_ = []
            start_dim = 0
            index_group = indexes[1][0]
            for ci in indexes[0]:
                cis_.append(ci + start_dim)
                start_dim += ci.shape[0]
            cis = [None for _ in range(index_group.shape[0])]
            for i, idx in enumerate(index_group):
                cis[idx] = cis_[i]
            index = torch.concat(cis, axis=0)
    if index is None:
        index = indexes[0][0]
        if w.shape[axis] == index.shape[0] * 2:
            print(
                f"weight shape {w.shape[axis]}, index shape {index.shape[0]}. Will double index shape"
            )
            index = torch.concat([index, index + index.shape[0]])
    index = index.to(w).long()
    w_ = torch.zeros_like(w).to(w)
    m_ = torch.zeros_like(w).to(w)
    wa_index = torch.arange(index.shape[0]).to(w).long()

    # real_len = int(round(index.shape[0]/1.2))
    # cond = index<real_len
    # cond[real_len:]=False
    # print("raw len",real_len,"index len",index.shape[0])
    # if not (wa_index[cond]==index[cond]).all():
    #     no_equal = wa_index[cond] != index[cond]
    #     print("raw",wa_index[cond],"index",index[cond])
    #     print("false match",wname,axis,"raw",wa_index[cond][no_equal],"index",index[cond][no_equal])
    # if (anno is not None and any(a in anno[0]
    #                              for a in ["qkv", "group"])) or
    # if p == 48:
    #     if p == 48:
    #         print(wname, axis, index.shape, w_.shape, w.shape)
    #         # index = index.sort(descending=True).values
    #     w_.index_copy_(axis, index, w.index_select(axis, wa_index))
    #     # if "blocks.11.mlp.fc2" in wname:
    #     #     print("set zero")
    #     #     w_.fill_(0)
    #     # if "norm" in wname:
    #     #     w_.fill_(1)
    #     # if "head" in wname:
    #     #     print("head",index)

    # else:
    #     w_.index_copy_(axis, wa_index, w.index_select(axis, wa_index))
    # print(wname, axis, index.shape, index)
    w_.index_copy_(axis, index, w.index_select(axis, wa_index))
    if m is not None:
        m_.index_copy_(axis, index, m.index_select(axis, wa_index))
    else:
        m_.index_copy_(axis, index, ma.index_select(axis, wa_index))
    if ma is not None:
        m_[~ma.bool()] = 0
    return w_, m_, wa


def dissimilarity(w_b, w_a):
    w_a = torch.clone(w_a)
    w_b = torch.clone(w_b)
    n = w_a.shape[0]
    w_a = w_a.reshape(n, -1)
    w_b = w_b.reshape(n, -1)

    wa_mask = w_a.bool()
    wb_mask = w_b.bool()
    w_a[~wb_mask] = 0
    w_b[~wa_mask] = 0
    w_a_norm = torch.norm(w_a, dim=1, keepdim=True)
    w_b_norm = torch.norm(w_b, dim=1, keepdim=True)
    scal = (w_b_norm @ w_a_norm.T)
    # print(w_b.shape, w_a.shape)

    return (w_b @ w_a.T), scal

    # w_a = w_a / w_a_norm
    # w_b = w_b / w_b_norm

    # return (w_b @ w_a.T), torch.tensor(1).to(w_a)


def weight_match(modela: nn.Module,
                 modelb: nn.Module,
                 perm,
                 max_iter=100,
                 random_state=0,
                 tol=5,
                 shuffle_first_iter=False,
                 overlap=1,
                 dim_tol=None):
    perm = deepcopy(perm)
    for items in perm.values():
        for v in items:
            if len(v) == 2:
                v.append(None)
    perm_mats = {}
    perm_max = {}
    perm_max_qk = {}
    axis2perm = {}
    for k, v in perm.items():
        for axis, w, anno in v:
            if w not in axis2perm:
                axis2perm[w] = []
            axis2perm[w].append([axis, k, anno])

    params_a = {k: v.cpu() for k, v in modela.state_dict().items()}
    params_b = {k: v.cpu() for k, v in modelb.state_dict().items()}
    perm_names = list(perm.keys())
    no_progress_count = 0
    generator = torch.Generator()
    generator.manual_seed(random_state)
    for iter in range(max_iter):
        progress = False
        total_oldL = 0
        total_newL = 0
        for p_ix in torch.randperm(len(perm_names), generator=generator):
            p = perm_names[p_ix]
            A = None
            A_shadow = None
            A_scale = None
            Agroup = None
            Agroup_scale = None
            Aqk = None
            Aqk_shadow = None
            Aqk_scale = None
            for axis, wk, anno in perm[p]:
                w_a = params_a[wk]
                w_b = params_b[wk]
                for a_, p_, anno_ in axis2perm[wk]:
                    if a_ == axis:
                        continue
                    elif p_ in perm_mats:
                        # print(wk, a_, anno_)
                        perm_mat = perm_mats[p_]
                        w_b, _, w_a = _perm(w_b,
                                            a_,
                                            perm_mat,
                                            anno=anno_,
                                            wa=w_a)
                # print(w_a.shape,w_b.shape,axis,wk)
                w_a = torch.moveaxis(w_a, axis, 0)
                w_b = torch.moveaxis(w_b, axis, 0)

                if anno is not None:
                    anno_name, anno_params = anno
                    if "ignore" in anno_name:
                        continue
                    elif "qkv" in anno_name:
                        group, = anno_params
                        head_dim = w_a.shape[0] // (3 * group)
                        w_a = w_a.reshape(3, group, head_dim, -1)
                        w_b = w_b.reshape(3, group, head_dim, -1)
                        if A is None:
                            A = [None for _ in range(group)]
                            A_scale = [None for _ in range(group)]
                            Aqk = [None for _ in range(group)]
                            Aqk_scale = [None for _ in range(group)]
                            Agroup = [None]
                            Agroup_scale = [None]
                        for i in range(2):
                            for g in range(group):
                                A_, A_scal_ = dissimilarity(
                                    w_b[i, g], w_a[i, g])
                                Aqk[g] = A_ if Aqk[g] is None else (Aqk[g] +
                                                                    A_)
                                Aqk_scale[
                                    g] = A_scal_ if Aqk_scale[g] is None else (
                                        Aqk_scale[g] + A_scal_)

                        for g in range(group):
                            A_, A_scal_ = dissimilarity(w_b[2, g], w_a[2, g])
                            A[g] = A_ if A[g] is None else (A[g] + A_)
                            A_scale[g] = A_scal_ if A_scale[g] is None else (
                                A_scale[g] + A_scal_)

                        w_a = torch.moveaxis(w_a, 1, 0).reshape(group, -1)
                        w_b = torch.moveaxis(w_b, 1, 0).reshape(group, -1)
                        A_, A_scal_ = dissimilarity(w_b, w_a)
                        Agroup[0] = A_ if Agroup[0] is None \
                            else  (Agroup[0] +      A_)
                        Agroup_scale[0] = A_scal_ if Agroup_scale[0] is None \
                                else (Agroup_scale[0] + A_scal_)

                    elif "group" in anno_name:
                        group, = anno_params
                        head_dim = w_a.shape[0] // group
                        w_a = w_a.reshape(group, head_dim, -1)
                        w_b = w_b.reshape(group, head_dim, -1)
                        if A is None:
                            A = [None for _ in range(group)]
                            A_scale = [None for _ in range(group)]
                            Agroup = [None]
                            Agroup_scale = [None]
                        for g in range(group):
                            A_, A_scal_ = dissimilarity(w_b[g], w_a[g])
                            A[g] = A_ if A[g] is None else (A[g] + A_)
                            A_scale[g] = A_scal_ if A_scale[g] is None else (
                                A_scale[g] + A_scal_)
                        w_a = w_a.reshape(group, -1)
                        w_b = w_b.reshape(group, -1)
                        A_, A_scal_ = dissimilarity(w_b, w_a)
                        Agroup[0] = A_ if Agroup[0] is None \
                            else  (Agroup[0] + A_)
                        Agroup_scale[0] = A_scal_ if Agroup_scale[0] is None \
                                else (Agroup_scale[0] + A_scal_)
                    else:
                        raise NotImplementedError

                else:
                    if A is None:
                        A = [None]
                        A_scale = [None]
                    A_, A_scal_ = dissimilarity(w_b, w_a)
                    A[0] = A_ if A[0] is None else (A[0] + A_)
                    A_scale[0] = A_scal_ if A_scale[0] is None else (
                        A_scale[0] + A_scal_)
                # print(axis, wk, anno)
            # print(len(A), "lena")
            A = [a / s.clamp_min(1e-8) for a, s in zip(A, A_scale)]
            A_shadow = [None for _ in range(len(A))]
            if p in perm_max:
                max_values = perm_max[p]
                for z in range(len(max_values)):
                    new_max = A[z].max() + 1e-6
                    if new_max > max_values[z]:
                        max_values[z] = new_max
            else:
                max_values = [a.max() + 1e-6 for a in A]
            perm_max[p] = max_values
            cis_all = []
            cis = []

            for ai in range(len(A)):
                A_ = A[ai]
                if dim_tol is not None:
                    K = int(dim_tol // len(A))
                    assert K * len(
                        A
                    ) == dim_tol, "dim_tol must be a multiple of num heads"
                elif overlap is not None:
                    K = int(A_.shape[0] * (1 - overlap) + 0.5)
                else:
                    raise NotImplementedError
                A_shadow[ai] = F.pad(A_, (0, K, 0, K),
                                     mode="constant",
                                     value=0)
                A_ = F.pad(A_, (0, K, 0, K),
                           mode="constant",
                           value=max_values[ai])
                A[ai] = A_

                ri, ci = linear_sum_assignment(A_.detach().numpy(),
                                               maximize=True)
                assert (torch.tensor(ri) == torch.arange(len(ri))).all()
                ci = torch.tensor(ci)
                cis.append(ci)
            cis_all.append(cis)
            cigroup = None
            if Agroup is not None:
                cigroup = []
                Agroup = [
                    a / s.clamp_min(1e-8)
                    for a, s in zip(Agroup, Agroup_scale)
                ]
                for A_ in Agroup:
                    ri, ci = linear_sum_assignment(A_.detach().numpy(),
                                                   maximize=True)
                    assert (torch.tensor(ri) == torch.arange(len(ri))).all()
                    ci = torch.tensor(ci)
                    cigroup.append(ci)
                cis_all.append(cigroup)
            cisqk = None
            if Aqk is not None:
                cisqk = []
                Aqk = [a / s.clamp_min(1e-8) for a, s in zip(Aqk, Aqk_scale)]
                Aqk_shadow = [None for _ in range(len(Aqk))]
                if p in perm_max_qk:
                    max_values = perm_max_qk[p]
                    for z in range(len(max_values)):
                        new_max = Aqk[z].max() + 1e-6
                        if new_max > max_values[z]:
                            max_values[z] = new_max
                else:
                    max_values = [a.max() + 1e-6 for a in Aqk]
                perm_max_qk[p] = max_values
                for ai in range(len(Aqk)):
                    A_ = Aqk[ai]
                    if dim_tol is not None:
                        K = int(dim_tol // len(A))
                        assert K * len(
                            A
                        ) == dim_tol, "dim_tol must be a multiple of num heads"
                    elif overlap is not None:
                        K = int(A_.shape[0] * (1 - overlap) + 0.5)
                    else:
                        raise NotImplementedError
                    Aqk_shadow[ai] = F.pad(A_, (0, K, 0, K),
                                           mode="constant",
                                           value=0)
                    A_ = F.pad(A_, (0, K, 0, K),
                               mode="constant",
                               value=max_values[ai])
                    Aqk[ai] = A_
                    ri, ci = linear_sum_assignment(A_.detach().numpy(),
                                                   maximize=True)
                    assert (torch.tensor(ri) == torch.arange(len(ri))).all()
                    ci = torch.tensor(ci)
                    cisqk.append(ci)
                cis_all.append(cisqk)

            eye_ = None
            oldL = 0
            newL = 0
            old_p = None
            if p in perm_mats:
                old_p = perm_mats[p]
            # for A_, ci in zip(A_shadow, cis):
            for ai, (A_, ci) in enumerate(zip(A, cis)):
                if eye_ is None:
                    eye_ = torch.eye(ci.shape[0])
                if old_p is not None:
                    oldL += torch.einsum('ij,ij->i', A_,
                                         eye_[old_p[0][ai]]).sum().item()
                newL += torch.einsum('ij,ij->i', A_, eye_[ci]).sum().item()
            if Agroup is not None:
                eye_group = None
                for ai, (A_, ci) in enumerate(zip(Agroup, cigroup)):
                    if eye_group is None:
                        eye_group = torch.eye(ci.shape[0])
                    if old_p is not None:
                        oldL += torch.einsum('ij,ij->i', A_,
                                             eye_[old_p[1][ai]]).sum().item()
                    newL += torch.einsum('ij,ij->i', A_,
                                         eye_group[ci]).sum().item()
            if Aqk is not None:
                # for A_, ci in zip(Aqk_shadow, cisqk):
                for ai, (A_, ci) in enumerate(zip(Aqk_shadow, cisqk)):
                    if old_p is not None:
                        oldL += torch.einsum('ij,ij->i', A_,
                                             eye_[old_p[2][ai]]).sum().item()
                    newL += torch.einsum('ij,ij->i', A_, eye_[ci]).sum().item()
            if shuffle_first_iter and iter == 0:
                for cis in cis_all:
                    for ci in cis:
                        if ci is not None:
                            idx = torch.randperm(ci.numel(),
                                                 generator=generator)
                            ci[:] = ci[idx]
                newL = 0
            progress = progress or newL > oldL
            total_oldL += oldL
            total_newL += newL
            perm_mats[p] = cis_all
        if not progress:
            no_progress_count += 1
            if no_progress_count >= tol:
                break
        else:
            no_progress_count = 0
        print("iter", iter, "process count", no_progress_count, "oldL",
              total_oldL, "newL", total_newL)
    return perm_mats, axis2perm


def apply_perm(params_a,
               params_b,
               perm_mats,
               axis2perm,
               keep_unpermutated=True):
    params_b_match = {}
    params_a_adapt = {}
    params_mask = {}
    for wk, v in axis2perm.items():
        w_b = params_b[wk]
        w_a = params_a[wk]
        m_ = None
        for axis, p, anno in v:
            perm_mat = perm_mats[p]
            w_b, m_, w_a = _perm2(w_b, axis, perm_mat, m_, anno, w_a, wk, p)
            # w_b, m_, w_a = _perm(w_b, axis, perm_mat, m_, anno, w_a)
        params_b_match[wk] = w_b
        params_a_adapt[wk] = w_a
        params_mask[wk] = m_

    for k, v in params_b.items():
        if k not in params_b_match:
            print(f"not permutated weight b {k} (keep {keep_unpermutated})")
            if keep_unpermutated:
                params_b_match[k] = v
    for k, v in params_a.items():
        if k not in params_a_adapt:
            print(f"not permutated weight a {k} (keep {keep_unpermutated})")
            if keep_unpermutated:
                params_a_adapt[k] = v
    return params_a_adapt, params_b_match, params_mask


@torch.no_grad()
def network_adapt(net: nn.Module, perm_mats, axis2perm):
    modules = {k: v for k, v in net.named_modules()}
    for wk, v in axis2perm.items():
        wk_path = wk.split(".")
        modules_name = ".".join(wk_path[:-1])
        params_name = wk_path[-1]
        m = modules[modules_name]
        w: nn.Parameter = getattr(m, params_name)

        for axis, p, anno in v:
            perm_mat = perm_mats[p]
            K = inference_K(w, axis, anno, perm_mat)
            new_w = w
            new_w = padding_weight(w, axis, anno, K)
            w.set_(new_w)
            if anno is not None:
                if "qkv" in anno[0]:
                    group = anno[1][0]
                    target_dim = w.shape[axis]
                    head_dim = target_dim // (group * 3)
                    attn_module_name = ".".join(wk_path[:-2])
                    attn_module = modules[attn_module_name]
                    attn_module.head_dim = head_dim
            if params_name == "weight":
                if isinstance(m, nn.Linear):
                    m.out_features, m.in_features = new_w.shape[
                        0], new_w.shape[1]
                elif isinstance(m, nn.Conv2d):
                    m.out_channels, m.in_channels = new_w.shape[
                        0], new_w.shape[1]
                elif isinstance(m, nn.LayerNorm):
                    m.normalized_shape = (new_w.shape[0], )
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = new_w.shape[0]
                else:
                    print(type(m))
                    raise NotImplementedError


def avg_param_partial(params_a_, params_b_, params_mask_):
    param_ = {}
    for k, v in ((k, v) for k, v in params_a_.items()):

        if k in params_mask_:
            param_[k] = (params_b_[k] + v) / (params_mask_[k] + 1)
        else:
            param_[k] = (params_b_[k] + v) / 2

    return param_