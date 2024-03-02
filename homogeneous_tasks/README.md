# Merging Models of Homogeneous Tasks

## Data

3 datasets were used in this experiment:

- CIFAR-10: Automatic download with torchvision
- CIFAR-100: Automatic download with torchvision
- ImageNet: Downloadable from https://image-net.org/download.php

## Run

```shell
cd ./homogeneous_tasks
```

**Training models** 

```shell
python -m training_script.cifar_resnet20
```

**Evaluating merging methods**  

- Evaluating the basic performance (i.e. the original models, ensemble of the original models)

```shell
python -m base_model_concept_merging --config-name=cifar50_resnet20
```

- Evaluating the merging method

```shell
python -m mudsc_concept_merging --config-name=cifar50_resnet20 --suffix=$SUFFIX
```

- Note that `$SUFFIX` can be:

  > **_avg:** Direct average of the original models
  >
  > **_act:** An equivalent implementation of [Zipit](https://github.com/gstoica27/ZipIt) without partial zip. For the models without group structure (i.e. ViT, ResnetGN), we test them with the original Zipit. For the model with group structure (i.e. ViT, ResnetGN), we test them with our implementation.
  >
  > **_act_useperm:** Activation-based alignment (A. Align)
  >
  > **_useperm:** An equivalent implementation of [Git Rebasin](https://github.com/samuela/git-re-basin). For the models without group structure, we test them with a [pytorch implement of Git Rebasin](https://github.com/themrzmaster/git-re-basin-pytorch). For the model with group structure (i.e. ViT, ResnetGN), we test them with our implementation. 
  >
  > **""**: Weight-based Zip(W. Zip)
  >
  > **_act_iws_fs_useperm**: Alignment-based MuDSC
  >
  > **_act_iws_fs_useperm_train**: Alignment-based MuDSC  tested on train dataset (for searching balanced factor)
  >
  > **_act_iws_fs**: Zip-based MuDSC 
  >
  > **_act_iws_fs_train**: Zip-based MuDSC tested on train dataset (for searching balanced factor)

  