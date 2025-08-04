import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_provider.data_loader import UEAloader
from torch.utils.data import Dataset
from utils.tools import plot_signal


def plot_results(signal, cA3, cD3, cD2, cD1):
    # 绘制原始信号
    plt.figure(figsize=(12, 10))
    plt.subplot(6, 1, 1)
    plt.plot(signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # 绘制第三级近似系数
    plt.subplot(6, 1, 2)
    plt.plot(cA3)
    plt.title('Approximation Coefficients (Level 3)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # 绘制第三级细节系数
    plt.subplot(6, 1, 3)
    plt.plot(cD3)
    plt.title('Detail Coefficients (Level 3)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # 绘制第二级细节系数
    plt.subplot(6, 1, 4)
    plt.plot(cD2)
    plt.title('Detail Coefficients (Level 2)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # 绘制第一级细节系数
    plt.subplot(6, 1, 5)
    plt.plot(cD1)
    plt.title('Detail Coefficients (Level 1)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # 绘制重构信号
    # plt.subplot(6, 1, 6)
    # plt.plot(reconstructed_signal)
    # plt.title('Reconstructed Signal')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def dwt_transform(signal, level, dwt_func):
    # 选择小波函数，这里使用 db4 小波
    wavelet = dwt_func
    dim = signal.shape[-1]
    signal_dwt = [pywt.wavedec(signal[:, :, d], wavelet, level=level) for d in range(dim)]
    # n = 1
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(pywt.waverec(signal_dwt[0], wavelet)[n,:-1],c='r')
    # plt.plot(signal[n,:,0].numpy()+1,c='g')
    # plt.subplot(3, 1, 2)
    # plt.plot(signal_dwt[0][0][n],c='pink')
    # plt.subplot(3, 1, 3)
    # plt.plot(signal_dwt[0][1][n], c='black')
    # plt.savefig('level2.jpg')
    coeffs_sizes = [signal_dwt[0][i].shape[-1] for i in range(level+1)]
    batch = signal_dwt[0][0].shape[0]
    return_signal = []
    for l in range(level+1):
        coeff = np.empty([batch, coeffs_sizes[l], 0])
        for d in range(dim):
            coeff = np.concatenate((coeff, signal_dwt[d][l][:,:,np.newaxis]), axis=-1)
        return_signal.append(torch.from_numpy(coeff))
    # plot_signal(signal[0], 'original_hw.jpg')
    # plot_signal(return_signal[0][0], 'A_LSST.png')
    # plot_signal(return_signal[1][0], 'D_LSST.png')
    return return_signal


class LoadClssificationDataset(Dataset):
    def __init__(self, task_name, data_dir, split, config, **kwargs):
        if task_name == 'classification':
            data_path = data_dir
            data = UEAloader(root_path=data_path, flag=split)
            self.x = torch.from_numpy(data.feature_df.values.reshape(data.all_IDs.size, data.max_seq_len, -1)).type(torch.float32)
            self.y = torch.from_numpy(data.labels_df.values).type(torch.LongTensor)
            self.coeffs_size = [len(c) for c in pywt.wavedec(self.x[0, :, 0], config['data']['dwt_func'], level=config['data']['dwt_level'])]
            self.num_classes = data.class_names.size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


