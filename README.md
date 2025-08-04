# TSCMixer: A multi-view method for Multivariable Time Series Classification
# Overview
## 1.Config file
We have provided a config file at <kbd> ./config/TSCMixer.yaml</kbd>. This document consists of four parts: **Task Settings**, **Model Configuration**, **Training Configuration**, and **Dataset**.
- In the **Task Settings**, the parameter <kbd>model_name</kbd> only needs to be modified to the corresponding model name when changing the model. This parameter solely affects the filename of the model weights when they are saved.
- In the **Model Configuration**, the hyperparameters correspond one-to-one with those mentioned in the paper. Parameters <kbd>d_intermediate</kbd> and <kbd>ssm_cfg</kbd>, which are not mentioned in the paper, remain fixed in our experiments, they are included here only to reveal the key internal structure of the model.
- The **Model Configuration** and **Training Configuration** parameters corresponding to the experimental results in the paper are shown in the table below.

| Datasets   | epochs | batch size | learning rate | M   | N   | d    | $L_p$ | $D_{rep}$ |
|:------------:|:--------:|:------------:|:---------------:|:-----:|:-----:|:------:|:-------:|:-----------:|
| AWR        | 200    | 512        | 0.0001        | 2   | 6   | 128  | 32    | 32        |
| BM         | 100    | 64         | 0.0001        | 2   | 3   | 64   | 32    | 32        |
| CR         | 200    | 64         | 0.0001        | 2   | 3   | 128  | 32    | 32        |
| EP         | 200    | 256        | 0.0005        | 2   | 3   | 128  | 16    | 16        |
| FM         | 200    | 512        | 0.0001        | 2   | 6   | 64   | 16    | 64        |
| HMD        | 220    | 256        | 0.00003       | 2   | 2   | 32   | 32    | 32        |
| HW         | 100    | 256        | 0.0005        | 2   | 5   | 128  | 8     | 256       |
| HB         | 150    | 256        | 0.0001        | 2   | 2   | 64   | 32    | 32        |
| LB         | 300    | 256        | 0.0002        | 2   | 3   | 32   | 8     | 64        |
| MI         | 9      | 128        | 0.0002        | 2   | 3   | 32   | 16    | 16        |
| NATODPS    | 200    | 256        | 0.0002        | 1   | 3   | 32   | 16    | 32        |
| PD         | 300    | 1024       | 0.0003        | 1   | 3   | 32   | 4     | 32        |
| RS         | 100    | 256        | 0.0002        | 2   | 3   | 32   | 8     | 64        |
| SCP1       | 200    | 512        | 0.0001        | 2   | 3   | 64   | 128   | 32        |
| SCP2       | 100    | 512        | 0.0001        | 1   | 3   | 32   | 16    | 16        |
| UG         | 150    | 512        | 0.0004        | 2   | 4   | 64   | 128   | 64        | 
- In the **Dataset** section, only the corresponding dataset name needs to be modified before each run. Parameters <kbd>dwt_level</kbd> and <kbd>dwt_func</kbd> refer to the decomposition level and wavelet basis function, respectively, used in the discrete wavelet transform during data preprocessing to convert the original sequence into the time-frequency domain.
## 2.Datasets
We have provided 2 sample datasets under <kbd> ./dataset/classification/UEA/</kbd>. The whole UEA datasets can be downloaded from the [official website](https://www.timeseriesclassification.com/index.php/) in <kbd>.ts</kbd> format.
## 3.Model
- Under <kbd>./model/</kbd>, we have provided the proposed model in <kbd>TSCMixer.py</kbd>, as well as the models corresponding to the ablation experiments.

- Under <kbd>./layers/</kbd>, there are some modules used in our model; in addition, we have also provided other commonly used modules for exploration purposes.

- Under <kbd>./exp/</kbd>, we have provided the scripts for the training process.

## 4.Visualization
- In <kbd>./utils/tools.py</kbd>, we have provided a heatmap visualization function <kbd>heatmap(hidden_state, save_path)</kbd>.
# Installation

# Run
