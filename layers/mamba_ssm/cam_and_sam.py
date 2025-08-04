from torch import nn
import torch
from einops import rearrange
import torch.nn.functional as f


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.f1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b l d -> b d l")
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        out = rearrange(out, "b d l -> b l d")
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        x = rearrange(x, "b l d -> b d l")
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        x = rearrange(x, "b d l -> b l d")
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, ratio=16, kernel_size=3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        # out = nn.Dropout()
        return out


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim,choose):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.choose = choose

    def forward(self, x):
        if self.choose == 'variable':
            q = self.query(x).transpose(-2,-1)
            k = self.key(x).transpose(-2,-1)
            v = self.value(x).transpose(-2,-1)
        else:
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.output_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)
        output = torch.matmul(attn_weights, v)
        if self.choose == 'variable':
            output = output.transpose(-2,-1)
        return output


class ResidualBlock(nn.Module):
    """
    Args:
        hidden_size: input dimension (Channel) of 1d convolution
        output_size: output dimension (Channel) of 1d convolution
        kernel_size: kernel size
    """

    def __init__(self, input_size, output_size, kernel_sizes=[3, 3, 3]):
        super(ResidualBlock, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=output_size,
                               kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=output_size,
                               out_channels=output_size,
                               kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv1d(in_channels=output_size,
                               out_channels=output_size,
                               kernel_size=kernel_sizes[2])
        self.conv_skip = nn.Conv1d(in_channels=input_size,
                                   out_channels=output_size,
                                   kernel_size=1)

        self.norm1 = nn.BatchNorm1d(num_features=output_size)
        self.norm2 = nn.BatchNorm1d(num_features=output_size)
        self.norm3 = nn.BatchNorm1d(num_features=output_size)
        self.norm_skip = nn.BatchNorm1d(num_features=output_size)

    def forward(self, x):
        h = x
        h = f.pad(h, (int(self.kernel_sizes[0] / 2), int(self.kernel_sizes[0] / 2)), "constant", 0)
        h = f.relu(self.norm1(self.conv1(h)))

        h = f.pad(h, (int(self.kernel_sizes[1] / 2), int(self.kernel_sizes[1] / 2)), "constant", 0)
        h = f.relu(self.norm2(self.conv2(h)))

        h = f.pad(h, (int(self.kernel_sizes[2] / 2), int(self.kernel_sizes[2] / 2)), "constant", 0)
        h = self.norm3(self.conv3(h))

        s = self.norm_skip(self.conv_skip(x))
        h += s
        h = f.relu(h)
        return h

