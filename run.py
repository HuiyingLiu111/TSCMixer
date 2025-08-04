import os
import torch
import argparse
import random
import numpy as np
import yaml
from utils.tools import set_seed
from exp.exp_classification import Exp_Classification
from utils.wavelet_transfer import LoadClssificationDataset


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None


def print_configs(configs):
    for k in configs.keys():
        if isinstance(configs[k], dict):
            print(f"{k}:")
            for kk in configs[k].keys():
                print(f"  {kk}: {configs[k][kk]}")
        else:
            print(f"{k}: {configs[k]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/TSCMixer.yaml')
    parser.add_argument('--seed', type=int, default=3047, help='random seed')
    args = parser.parse_args()
    set_seed(args.seed)
    configs = load_config(args.config_path)

    # load data
    train_data = LoadClssificationDataset(task_name=configs['task_name'], data_dir=configs['data']['data_dir'], split='TRAIN', config=configs)
    valid_data = LoadClssificationDataset(task_name=configs['task_name'], data_dir=configs['data']['data_dir'], split='TRAIN', config=configs)
    test_data = LoadClssificationDataset(task_name=configs['task_name'], data_dir=configs['data']['data_dir'], split='TEST', config=configs)

    # Data information for print
    configs['data']['coeffs_size'] = train_data.coeffs_size
    configs['data']['seq_len'] = train_data.x.shape[1]
    configs['data']['num_dims'] = train_data.x.shape[2]
    configs['data']['num_classes'] = train_data.num_classes
    _, configs['data']['num_per_classes'] = torch.unique(train_data.y, return_counts=True)
    if not torch.cuda.is_available():
        configs['training']['use_gpu'] = False
    print_configs(configs)

    # model define
    # TODO load model parameters from a .yaml file
    Exp = Exp_Classification(configs)
    Exp.train(train_data, valid_data, test_data)
    # Exp.test(test_data)
    torch.cuda.empty_cache()


