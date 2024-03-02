# Merging Models of Heterogeneous Tasks

## Data and Models

**Models**

- Download the models from the [repository](https://github.com/alexsax/visual-prior/tree/networks/assets/pytorch) and put them under the `$USER_HOME$/.cache/torch/hub/checkpoints`.

**Data**

- Download the files of the data from [Taskonomy](https://github.com/StanfordVL/taskonomy/tree/master/data) and structure them as described below:

  ```shell
  taskonomy_data/
    ihlen/
      class_object/
      depth_euclidean/
      ...
    mcdade/
      class_object/
      depth_euclidean/
      ...
    ...
  ```

  - You can download the necessary files directly from the links in [data_links.txt](data_links.txt) as the official download tools always fail.

\* To loading data faster, we compressed the sample from 512\*512 to 256\*256 with `compress_dataset.py`.

## Run

```shell
cd ./heterogeneous_tasks
```

**Evaluate basic performance**

```shell
# Evaluate the performance with the average of labels
python -m eval.eval_avg_estimator.py 
# Evaluate the performance with the average of labels
python -m eval.eval_avg_model.py
# Evaluate the performance of pretrained model.
python -m eval.eval_pretrained_model.py
```

- For fair comparisons , we reset the batch normalization of the all pretrained models and merged models with specific partial training data.  Note that the performance of pretrained models after reset are still close to that before reset (referring to [basic performance](results_of_basic_performance.ipynb)).

**Evaluate with Git Rebasin**

```shell
# Generate Git Rabasin Encoder
python -m generate_encoder.rebasin_encoder
# Evaluate Git Rabasin Encoder
python -m eval.eval_rebasin_model
```

**Evaluate with Zipit**

```shell
# Generate Zipit Encoder
python -m generate_encoder.zipit_encoder
# Evaluate Zipit Encoder
python -m eval.eval_zipit_model
```

**Evaluate with MuDSC**

```shell
# Generate MuDSC Encoder
python -m generate_encoder.mudsc_encoder --suffix=$SUFFIX
# Evaluate MuDSC Encoder
python -m eval.eval_mudsc_model --suffix=$SUFFIX
```

- Note that `$SUFFIX` can be:

  > **_act_useperm:** Activation-based alignment (A. Align)
  >
  > **""**: Weight-based Zip(W. Zip)
  >
  > **_act_iws_fs_useperm**: Alignment-based MuDSC
  >
  > **_act_iws_fs**: Zip-based MuDSC 

**Calculate the scaled performance**

calculate the scaled performance with [calculate_scaled_performance.ipynb](calculate_scaled_performance.ipynb).