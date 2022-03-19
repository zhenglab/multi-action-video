# Implementation for m3a_relation_learner.
# 
# Code partially referenced from:
# https://github.com/tkipf/pygcn
# https://github.com/Megvii-Nanjing/ML-GCN
# https://github.com/jadore801120/attention-is-all-you-need-pytorch

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderSimplified(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dropout=0.1, 
                 layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderSimplified, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # No Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, src: Tensor) -> Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src

class TransformerDecoderSimplified(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dropout=0.1, 
                 layer_norm_eps=1e-5, batch_first=True,
                 kdim=None, vdim=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderSimplified, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               kdim=kdim, vdim=vdim, **factory_kwargs)
        # No Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, qsrc: Tensor, ksrc: Tensor, vsrc: Tensor) -> Tensor:
        src2 = self.cross_attn(qsrc, ksrc, vsrc)[0]
        src = qsrc + self.dropout(src2)
        src = self.norm(src)
        return src
