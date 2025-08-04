import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from utils.wavelet_transfer import dwt_transform
from layers.mamba_ssm.mixer_seq_simple import MixerModel as Mamba
from utils.tools import heatmap
import numpy as np
import math
import pywt
from layers.mamba_ssm.cam_and_sam import ChannelAttention,SpatialAttention
from layers.inception_time_pytorch.modules import InceptionModel


class ClassificationHead(nn.Module):
    def __init__(self, fusion_len, num_cls):
        super().__init__()
        self.fc = nn.Linear(fusion_len, num_cls)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs['data']['num_classes']
        self.dwt_func = configs['data']['dwt_func']
        self.dwt_level = configs['data']['dwt_level']
        self.patch_len = configs['model']['patch_len']
        self.Dicp = configs['model']['Dicp']
        self.N = configs['model']['N']
        self.dims = configs['data']['num_dims']
        self.d = configs['model']['d']
        self.M = configs['model']['M']
        self.d_intermediate = configs['model']['d_intermediate']
        self.ssm_cfg = configs['model']['ssm_cfg']
        self.device = torch.device('cuda:0') if configs['training']['use_gpu'] else torch.device('cpu')
        self._build_model()
        self._init_weight()

    def _build_model(self):
        self.Column_wise_Attention = ChannelAttention(self.d).to(self.device)
        self.Row_wise_Attention = nn.ModuleList([SpatialAttention()]).to(self.device)
        self.Mambas = nn.ModuleList([Mamba(d_model=self.d,
                                           n_layer=self.M,
                                           d_intermediate=self.d_intermediate,
                                           ssm_cfg=self.ssm_cfg,
                                           device=self.device,
                                           dtype=None)]).to(self.device)
        self.inceptions = InceptionModel(input_size=self.dims,
                                         num_classes=self.num_classes,
                                         filters=self.Dicp,
                                         depth=self.N).to(self.device)

        self.classifier = ClassificationHead(self.Dicp * 4 + self.d, self.num_classes).to(self.device)

        # # patching and embedding
        self.embeddings = nn.ModuleList([nn.Sequential(Rearrange('b c l d -> (b c) l d'), nn.Linear(self.patch_len, self.d, bias=False))]).to(self.device)

    def padding4patching(self, x):
        remainder = x.shape[1] % self.patch_len
        if remainder != 0:
            padding_length = self.patch_len - remainder
            x = rearrange(x, 'b l d -> b d l')
            x = F.pad(x, pad=(0, padding_length), mode='constant', value=0)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
        return None

    def forward(self, x):
        # -------------------------- TFL Stream ----------------------------------#
        x_dwt = dwt_transform(x, self.dwt_level, self.dwt_func)
        x_dwt = [((x_i-torch.mean(x_i, dim=1, keepdim=True).data)/torch.std(x_i, dim=1, keepdim=True).data).type(torch.float32) for x_i in x_dwt]

        x_d = x_dwt[0].to(self.device)
        x_d = self.padding4patching(x_d)
        x_d = x_d.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x_d = self.embeddings[0](x_d)

        # Variable Scan along Time (VST)
        x_d = rearrange(x_d, '(b c) l d -> b (c l) d', c=self.dims)

        x_d = x_d * self.Row_wise_Attention[0](x_d)

        X_RA_low = x_d
        X_RA_Mb_low = X_RA_low + self.Mambas[0](X_RA_low)
        X_RA_Mb_CA_low = X_RA_Mb_low * self.Column_wise_Attention(X_RA_Mb_low)
        H_TFL = torch.mean(X_RA_Mb_CA_low, dim=1)

        # --------------------------  TMS Stream ---------------------------------#
        H_TMS = self.inceptions(rearrange(x, 'b l d -> b d l').to(self.device))

        # Cross-domain feature fusion
        H_fusion = torch.concat([H_TFL, H_TMS], dim=1)

        out = self.classifier(H_fusion)
        return out

