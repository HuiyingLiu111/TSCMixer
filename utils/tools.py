import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import shuffle, choice
import subprocess
import seaborn as sns
plt.switch_backend('agg')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def get_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    lines = output.split('\n')

    for line in lines:
        if 'MiB /' in line:
            # print(line)

            gpu_usage = line.split('MiB / ')[0]
            gpu_usage = gpu_usage.split('|')[-1]
            gpu_usage = gpu_usage.strip()

            return (int(gpu_usage) - 10) / 1000


def heatmap(hidden_state, save_path):

    plt.figure()
    plt.rcParams['font.size'] = 8
    ax = sns.heatmap(hidden_state, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    # 设置x轴和y轴每隔100显示一个刻度
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.yaxis.set_major_locator(plt.MultipleLocator(8))

    ax.set_xticklabels([int(x) for x in ax.get_xticks()], rotation=0)
    ax.set_yticklabels([int(y) for y in ax.get_yticks()], rotation=0)
    # plt.title('heatmap')
    plt.savefig(save_path, dpi=300)


def plt_train_loss(loss_h_list, test_acc_list, save_path):
    # 创建图形和第一个 y 轴
    fig, ax1 = plt.subplots()
    # 绘制第一条曲线
    ax1.plot(loss_h_list, 'b-', label='train loss h')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('trainloss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # ax1.plot(loss_p_list, 'g-', label='train loss p')
    # ax1.set_xlabel('X axis')
    # ax1.set_ylabel('trainloss', color='g')
    # ax1.tick_params(axis='y', labelcolor='g')

    # 创建第二个 y 轴
    ax2 = ax1.twinx()
    # 绘制第二条曲线
    ax2.plot(test_acc_list, 'r-', label='test acc')
    ax2.set_ylabel('testacc', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    # 显示图例
    fig.legend(loc='upper right')
    plt.savefig(save_path)


def plot_signal(signal, sava_path):
    dims = signal.shape[-1]
    color_list = ['#EF767A', '#456990', '#48C0AA']
    plt.figure(figsize=(8,4))

    plt.rcParams['figure.facecolor'] = 'none'  # 画布透明
    plt.rcParams['axes.facecolor'] = 'none'  # 坐标轴透明
    for i in range(dims):
        plt.subplot(6, 1, i+1)
        s = signal[:, i]
        s = (s - torch.mean(s).data) / torch.std(s).data
        plt.plot(s, c=color_list[i], label=f'dim{i+1}')
        # plt.legend()
    # plt.xlabel('Time steps')
    # plt.ylabel('Amplitude')
    plt.savefig(sava_path, transparent=True, bbox_inches='tight')

