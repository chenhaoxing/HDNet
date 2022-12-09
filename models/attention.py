import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable, Function

class ECABlock(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

def min_max_norm(in_):
    """
        normalization
    :param in_:
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)

class SpatialGate(nn.Module):
    def __init__(self, in_dim=2, mask_mode='mask'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.mask_mode = mask_mode
        self.spatial = nn.Sequential(*[
            BasicConv(in_dim, in_dim, 3, 1, 1),
            BasicConv(in_dim, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2,  relu=False)
        ])
        self.act = nn.Sigmoid()
    def forward(self, x):
        x_compress = x
        x_out = self.spatial(x_compress)
        attention = self.act(x_out) 
        x = x * attention
        return x

    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, insn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.ins_norm = nn.InstanceNorm2d(out_planes, affine=False, track_running_stats=False) if insn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.ins_norm is not None:
            x = self.ins_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DualAttention(nn.Module):
    def __init__(self, gate_channels, mask_mode='mask'):
        super(DualAttention, self).__init__()
        self.ChannelGate = ECABlock(gate_channels)
        self.SpatialGate = SpatialGate(gate_channels, mask_mode=mask_mode)
        self.mask_mode = mask_mode
    def forward(self, x):
        x_ca = self.ChannelGate(x)
        x_out = self.SpatialGate(x_ca)
        return x_out + x_ca