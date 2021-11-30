# Implementation for m3a_helper.
# 
# Code partially referenced from:
# https://github.com/tkipf/pygcn
# https://github.com/Megvii-Nanjing/ML-GCN
# https://github.com/jadore801120/attention-is-all-you-need-pytorch

import os
import torch
import torch.nn as nn
import numpy as np
import math
import yaml
import random
import pickle
from torch.nn import Parameter
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
            
def get_graph_adj_matrix(num_classes, t, adj_file):
    with open(adj_file, 'rb') as f:
        graph_adj = torch.from_numpy(pickle.load(f))
        graph_adj = torch.where(graph_adj < t, torch.tensor(0, dtype=graph_adj.dtype), torch.tensor(1, dtype=graph_adj.dtype))
        
        graph_adj = graph_adj * 0.25 / (graph_adj.sum(0, keepdims=True) + 1e-6)
        graph_adj = graph_adj + torch.eye(num_classes)
        return graph_adj

def get_graph_adj_input(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    return torch.matmul(torch.matmul(A, D).t(), D)

def get_audio_lexicon(lexicon_file):
    with open(lexicon_file, 'rb') as f:
        return pickle.load(f)

def get_text_lexicon(lexicon_file):
    with open(lexicon_file, 'rb') as f:
        return torch.from_numpy(pickle.load(f))

def get_audio_embedding(audio_lexicon, batch_size):
    random.seed(1)
    label_dict = audio_lexicon['label']
    audio_dict = audio_lexicon['audio']

    audio_vec_list = []
    for i in range(batch_size):
        # label count random audio file
        random_audio_file_list = []
        for j in range(len(label_dict)):
            audio_num = len(label_dict[j])
            audio_file_list = []
            for k in range(1):
                audio_index = random.randint(0, audio_num - 1)
                audio_file_list.append(label_dict[j][audio_index])
            random_audio_file_list.append(audio_file_list)
        
        # label count audio vec
        random_audio_vec_list = []
        for audio_file_list in random_audio_file_list:
            audio_feature_list = []
            for audio_file in audio_file_list:
                audio_feature_list.append(audio_dict[audio_file])
            audio_feature_cat = np.concatenate(audio_feature_list, axis=0)
            random_audio_vec_list.append(audio_feature_cat)
        
        audio_vec_list.append(random_audio_vec_list)
    return torch.Tensor(audio_vec_list).float().cuda()
    
class M3AGraph(nn.Module):
    def __init__(self, cfg, feature_size):
        super(M3AGraph, self).__init__()

        num_classes = cfg.MODEL.NUM_CLASSES
        mmit_version = cfg.DATA.MMIT_VERSION
        
        self.cfg = cfg
        self.m3a_graph = nn.ModuleDict()
        
        graph_adj_t = 0.4
        graph_adj_file = os.path.join(
            cfg.DATA.PATH_TO_DATA_DIR, "mmit-m3a", "graph-adj-{}.pkl".format(cfg.DATA.MMIT_VERSION))
        self.graph_adj = get_graph_adj_matrix(num_classes, graph_adj_t, graph_adj_file).float().cuda()
        
        if "AUDIO" in cfg.M3A.MODAL_TYPE:
            lexicon_file = os.path.join(
                cfg.DATA.PATH_TO_DATA_DIR, "mmit-m3a", "audio-lexicon-{}.pkl".format(cfg.DATA.MMIT_VERSION))
            self.audio_lexicon = get_audio_lexicon(lexicon_file)
        
        if "TEXT" in cfg.M3A.MODAL_TYPE:
            lexicon_file = os.path.join(
                cfg.DATA.PATH_TO_DATA_DIR, "mmit-m3a", "text-lexicon-{}.pkl".format(cfg.DATA.MMIT_VERSION))
            self.text_lexicon = get_text_lexicon(lexicon_file).float().cuda()
        
        graph_param = {}
        graph_param["RGB"] = (feature_size, cfg.M3A.HIDDEN_LAYER, num_classes)
        graph_param["AUDIO"] = (cfg.M3A.AUDIO_VEC_SIZE, cfg.M3A.HIDDEN_LAYER, feature_size)
        graph_param["TEXT"] = (cfg.M3A.TEXT_VEC_SIZE, cfg.M3A.HIDDEN_LAYER, feature_size)
        graph_param["AUDIO_TEXT"] = (cfg.M3A.AUDIO_VEC_SIZE + cfg.M3A.TEXT_VEC_SIZE,
                                     cfg.M3A.HIDDEN_LAYER, feature_size)
        
        self.modal_joint_type = cfg.M3A.MODAL_JOINT_TYPE
        self.graph_list = cfg.M3A.MODAL_TYPE.split("_")
        if len(self.graph_list) == 1:
            if self.modal_joint_type == "NONE":
                graph_type = self.graph_list[0]
                self.build_graph(graph_type, graph_param)
            else:
                raise RuntimeError("Invalid Joint Type.")
        else:
            if self.modal_joint_type == "SUM":
                for graph_type in self.graph_list:
                    self.build_graph(graph_type, graph_param)
            elif self.modal_joint_type == "CAT" and "AUDIO_TEXT" in cfg.M3A.MODAL_TYPE:
                if "RGB" in cfg.M3A.MODAL_TYPE:
                    self.build_graph("RGB", graph_param)
                self.build_graph("AUDIO_TEXT", graph_param)
            else:
                raise RuntimeError("Invalid Joint Type.")

    def build_graph(self, graph_type, graph_param):
        dim_in = graph_param[graph_type][0]
        dim_inner = graph_param[graph_type][1]
        dim_out = graph_param[graph_type][2]
        
        # 2-layer GCN
        self.m3a_graph["{}_1".format(graph_type)] = GraphConvolution(dim_in, dim_inner, bias=False)
        self.m3a_graph["{}_relu".format(graph_type)] = nn.LeakyReLU(0.2)
        self.m3a_graph["{}_2".format(graph_type)] = GraphConvolution(dim_inner, dim_out, bias=False)
        
    def run_graph(self, graph_adj, x_input, graph_type):
        x_out = self.m3a_graph["{}_1".format(graph_type)](x_input, graph_adj)
        x_out = self.m3a_graph["{}_relu".format(graph_type)](x_out)
        x_out = self.m3a_graph["{}_2".format(graph_type)](x_out, graph_adj)
        return x_out
    
    def process_m3a_graph(self, x_input, graph_type):
        graph_adj = get_graph_adj_input(self.graph_adj).detach()
        if graph_type == "RGB":
            x_rgb = torch.unsqueeze(x_input, 1)
            x_rgb = x_rgb.expand(x_rgb.shape[0], self.graph_adj.shape[0], x_rgb.shape[2])
            x_rgb = self.run_graph(graph_adj, x_rgb, graph_type)
            x_out = x_rgb.mean(1)
        elif graph_type == "AUDIO":
            x_audio = get_audio_embedding(self.audio_lexicon, 1)
            x_audio = x_audio.view(x_audio.shape[1], x_audio.shape[2])
            x_audio = self.run_graph(graph_adj, x_audio, graph_type)
            x_audio = x_audio.transpose(0, 1)
            x_out = torch.matmul(x_input, x_audio)
        elif graph_type == "TEXT":
            x_text = self.text_lexicon
            x_text = self.run_graph(graph_adj, x_text, graph_type)
            x_text = x_text.transpose(0, 1)
            x_out = torch.matmul(x_input, x_text)
        elif graph_type == "AUDIO_TEXT":
            x_audio = get_audio_embedding(self.audio_lexicon, 1)
            x_audio = x_audio.view(x_audio.shape[1], x_audio.shape[2])

            x_text = self.text_lexicon
            
            x_audio_text = torch.cat((x_audio, x_text), 1).cuda()
            x_audio_text = self.run_graph(graph_adj, x_audio_text, graph_type)
            x_audio_text = x_audio_text.transpose(0, 1)
            
            x_out = torch.matmul(x_input, x_audio_text)
        return x_out

    def forward(self, x):
        x_out_list = []
        
        if len(self.graph_list) == 1:
            if self.modal_joint_type == "NONE":
                graph_type = self.graph_list[0]
                x_out_list.append(self.process_m3a_graph(x, graph_type))
        else:
            if self.modal_joint_type == "SUM":
                for graph_type in self.graph_list:
                    x_out_list.append(self.process_m3a_graph(x, graph_type))
            elif self.modal_joint_type == "CAT" and "AUDIO_TEXT" in self.cfg.M3A.MODAL_TYPE:
                if "RGB" in self.cfg.M3A.MODAL_TYPE:
                    x_out_list.append(self.process_m3a_graph(x, "RGB"))
                x_out_list.append(self.process_m3a_graph(x, "AUDIO_TEXT"))
        
        x_out = sum(x_out_list)
        return x_out

class M3AHead(nn.Module):
    def __init__(self, cfg, feature_size, pool_size):
        super(M3AHead, self).__init__()
        
        self.cfg = cfg
        self.avgpool = nn.AvgPool3d(pool_size, stride=(1, 1, 1), padding=(0, 0, 0))
        
        if cfg.M3A.MODE == "GRAPH":
            self.m3a_graph = M3AGraph(cfg = cfg, feature_size=feature_size)
        elif cfg.M3A.MODE == "TRSFMR":
#             self.m3a_trsfmr = M3ATransformer(cfg = cfg, feature_size=feature_size)
            raise NotImplementedError(
                "M3ATransformer is coming soon."
            )

        act_func = cfg.MODEL.HEAD_ACT
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        x = self.avgpool(x)
        
        if not self.training:
            x = x.mean([2, 3, 4])

        x = x.view(x.shape[0], -1)
        
        if self.cfg.M3A.MODE == "GRAPH":
            x = self.m3a_graph(x)
#         elif self.cfg.M3A.MODE == "TRSFMR":
#             x = self.m3a_trsfmr(x)

        if not self.training:
            x = self.act(x)
        
        return x
