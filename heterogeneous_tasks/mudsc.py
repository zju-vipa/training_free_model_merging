import torch
from torch import nn
from copy import deepcopy, copy
import pdb
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy
IGNORE_CFG = {"ignore": None}
IN_CFG = {"in": None}


def get_resnet_perm(block=3,
                    num_blocks=[3, 4, 6, 3],
                    use_bn_stat=False,
                    ingore_in=False,
                    scaled_bn_mean=False):
    bn_stat_cfg = {} if use_bn_stat else IGNORE_CFG
    if use_bn_stat:
        print("use bn")

    bn_mean_cfg = bn_stat_cfg
    if scaled_bn_mean:
        print("scale bn mean")
        bn_mean_cfg = {"scaled_bn": None}
    perm = {}
    cur_perm = 0
    perm[cur_perm] = [
        [0, "conv1.weight"],
        [0, "bn1.weight"],
        [0, "bn1.bias"],
        [0, "bn1.running_mean", bn_mean_cfg],
        [0, "bn1.running_var", bn_stat_cfg],
    ]

    res_perm = cur_perm
    cur_perm += 1

    in_config = {**IN_CFG, **IGNORE_CFG} if ingore_in else IN_CFG
    if ingore_in:
        print("ingore in")
    for l, block_num in enumerate(num_blocks):
        layer_id = l + 1

        for b in range(block_num):
            for c in range(1, block + 1):
                if b == 0 and c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", in_config],
                        [
                            1, f"layer{layer_id}.{b}.downsample.0.weight",
                            in_config
                        ],
                    ])

                elif c == 1:
                    perm[res_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", in_config],
                    ])
                else:
                    perm[cur_perm].extend([
                        [1, f"layer{layer_id}.{b}.conv{c}.weight", in_config],
                    ])
                    cur_perm += 1
                t = [
                    [0, f"layer{layer_id}.{b}.conv{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.weight"],
                    [0, f"layer{layer_id}.{b}.bn{c}.bias"],
                    [
                        0, f"layer{layer_id}.{b}.bn{c}.running_mean",
                        bn_mean_cfg
                    ],
                    [0, f"layer{layer_id}.{b}.bn{c}.running_var", bn_stat_cfg],
                ]
                if c < block:
                    perm[cur_perm] = t
                else:
                    if b == 0:
                        res_perm = cur_perm
                        cur_perm += 1
                        perm[res_perm] = [
                            [0, f"layer{layer_id}.{b}.downsample.0.weight"],
                            [0, f"layer{layer_id}.{b}.downsample.1.weight"],
                            [0, f"layer{layer_id}.{b}.downsample.1.bias"],
                            [
                                0,
                                f"layer{layer_id}.{b}.downsample.1.running_mean",
                                bn_mean_cfg
                            ],
                            [
                                0,
                                f"layer{layer_id}.{b}.downsample.1.running_var",
                                bn_stat_cfg
                            ],
                        ]
                    perm[res_perm].extend(t)

    perm[res_perm].extend([
        [1, "compress1.weight", in_config],
    ])
    return perm


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
        # 以每个节点最后一个输入模块作为统计激活值的模块
        weights_num = len(o["weights"])
        weights = o["weights"]
        for i in range(weights_num - 1, -1, -1):
            axis, w, cfg = weights[i]
            if "in" in cfg:
                cfg_in = cfg["in"]
                if cfg_in is None:
                    cfg_in = {}
                if isinstance(cfg["in"], str):
                    cfg_in = dict(name=cfg_in, dim=1)
                if "name" not in cfg_in:
                    cfg_in["name"] = ".".join(w.split(".")[:-1])
                if "dim" not in cfg_in:
                    cfg_in["dim"] = 1
                o["act_module"] = cfg_in
                break
        perm[p] = o
    for p, o in perm.items():
        if "in" not in o:
            o["in"] = []
        for axis, w, cfg in o["weights"]:
            if "in" not in cfg and w in in_perm:
                o["in"].extend(in_perm[w])
    return perm, weight_cfg


def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx + 1:]], dim=-1)


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
        # print("feats cat",feats.shape)
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
                 use_permute=False) -> None:
        self.reduce = reduce
        self.a = a
        self.b = b
        self.relax = relax
        self.iter = iter
        self.fix_sims = fix_sims
        self.sims_dict = None
        self.fix_rate = fix_rate
        if fix_sims:
            print(f"Fix Similarity {fix_rate}")
        self.compute_correlation = self._cossim if use_cos else self._correlation
        if use_cos:
            print("use cosine similarity")
        if no_fusion:
            self.fusion_weight = self.no_fusion_weight
            print("nofusion")
        if use_permute:
            self.fusion_weight = self.permute_weight
            print("permute weight")

    def transform(self, nets, perm, act_loader=None, in_weight_space=False):

        self.fusion_model: nn.Module = deepcopy(nets[0])
        perm = deepcopy(perm)
        self.perm, self.weight_cfg = refine_perm(perm)
        self.perm_names = list(perm.keys())
        self.perm_mats = {}
        self.params = [{k: v
                        for k, v in net.state_dict().items()} for net in nets]
        if act_loader is not None:
            self.act_transform(nets, act_loader)
        in_weight_space = in_weight_space or act_loader is None
        if in_weight_space:
            self.iter_transform()

        self.network_adapt(self.fusion_model)

        return self.get_merged_state_dict()

    def gen_act_sims(self, nets, act_loader: DataLoader):
        sims_dict = {k: CovarianceMetric() for k in self.perm}
        device = list(nets[0].parameters())[0].device
        hooks = []
        feats = [{} for _ in range(len(nets))]

        def add_hooks(net: nn.Module, idx):
            modules = {k: v for k, v in net.named_modules()}

            def prehook_gen(perm_name, act_dim):

                def prehook(m, x):
                    x = x[0].detach()
                    if act_dim != 1:
                        x = torch.moveaxis(x, act_dim, 1)
                    feats[idx][perm_name] = x
                    return None

                return prehook

            for k, o in self.perm.items():
                act_module = o["act_module"]
                act_module_name, act_dim = act_module["name"], act_module[
                    "dim"]
                module = modules[act_module_name]
                hooks.append(
                    module.register_forward_pre_hook(prehook_gen(k, act_dim)))

        def clear_hooks():
            for h in hooks:
                h.remove()

        for i, net in enumerate(nets):
            net.eval().cuda()
            add_hooks(net, i)
        for img, _ in tqdm(act_loader, desc="Computing activation"):
            img = img.cuda()
            for net in nets:
                net(img)
            for k, s in sims_dict.items():
                fs = [f[k].float() for f in feats]
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
            merge, unmerge, merge_value = self.fusion_weight(
                sims, r=self.reduce, a=self.a, b=self.b, get_merge_value=True)
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
                wvs = self.get_weight_vectors(p)
                sims = self.compute_correlation(wvs)
                if self.fix_sims and self.sims_dict is not None:
                    sims_act = self.sims_dict[p]

                    sims = sims * self.fix_rate + sims_act * (1 -
                                                              self.fix_rate)
                merge, unmerge, merge_value = self.fusion_weight(
                    sims,
                    r=self.reduce,
                    a=self.a,
                    b=self.b,
                    get_merge_value=True)
                merge = merge * len(self.params)
                # assert torch.diagonal(merge).all(), "no full"

                if merge_value > perm_state[p]:
                    progress = True
                perm_state[p] = merge_value

                self.perm_mats[p] = (merge, unmerge)
            total_newL = sum(v for v in perm_state.values()) / len(perm_state)
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
        # mats = [torch.zeros(Om, Om, device=device)]
        for i in range(1, N):
            try:
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    correlation[:Om, Om * i:Om * (i + 1)].cpu().numpy(),
                    maximize=True)
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
            merge_value = correlation[:Om, Om * i:Om *
                                      (i + 1)].cpu().numpy()[row_ind,
                                                             col_ind].mean()
            return merge, unmerge, merge_value
        return merge, unmerge

    def fusion_weight(
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
            best_idx = sims.reshape(-1).argmax()
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

    def no_fusion_weight(
        self,
        sims,
        r=.5,
        a=0,
        b=1,
        get_merge_value=False,
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
                    else:
                        print(type(m))
                        raise NotImplementedError

    @staticmethod
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
                else:
                    print(type(m))
                    raise NotImplementedError

    def get_merged_state_dict(self):
        merged_dict = {}
        rest_keys = set(self.params[0].keys())
        for wk, v in self.weight_cfg.items():
            rest_keys.remove(wk)
            ws = [param[wk] for param in self.params]

            for a, p, cfg in v:
                select = 1 if "in" in cfg else 0
                perm_mat = self.perm_mats[p][select]
                ws = self.perm_(a, ws, perm_mat, cfg)
            # w_final = sum(w for w in ws) / len(ws)
            merged_dict[wk] = ws

        for wk in rest_keys:
            print("no permutation weight", wk)
            ws = [param[wk] for param in self.params]
            # merged_dict[wk] = sum(w for w in ws) / len(ws)
            merged_dict[wk] = ws
        return merged_dict
