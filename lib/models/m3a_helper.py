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
from .m3a_relation_learner import GraphConvolution, PositionalEncoding, TransformerEncoderSimplified, TransformerDecoderSimplified
import torch.nn.functional as F

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

def get_audio_embedding(audio_lexicon, synonym_size, rng_seed, batch_size=1):
    random.seed(rng_seed)
    label_dict = audio_lexicon['label']
    audio_dict = audio_lexicon['audio']

    audio_vec_list = []
    for i in range(batch_size):
        # label count random audio file
        random_audio_file_list = []
        for j in range(len(label_dict)):
            audio_num = len(label_dict[j])
            audio_file_list = []
            for k in range(synonym_size):
                audio_index = random.randint(0, audio_num - 1)
                audio_file_list.append(label_dict[j][audio_index])
            random_audio_file_list.append(audio_file_list)
        
        # label count audio vec
        random_audio_vec_list = []
        for audio_file_list in random_audio_file_list:
            audio_feature_list = []

            # random select one audio in the lexicon
            audio_index = random.randint(0, synonym_size - 1)
            audio_file = audio_file_list[audio_index]
            audio_feature_list.append(audio_dict[audio_file])
#             # use all audios in the lexicon
#             for audio_file in audio_file_list:
#                 audio_feature_list.append(audio_dict[audio_file])

            audio_feature_cat = np.concatenate(audio_feature_list, axis=0)
            random_audio_vec_list.append(audio_feature_cat)
        
        audio_vec_list.append(random_audio_vec_list)
    return torch.Tensor(audio_vec_list).float().cuda()

class M3ARelationLearner(nn.Module):
    def __init__(self, cfg, nlayers, layer_dims):
        super(M3ARelationLearner, self).__init__()

        m3a_mode = cfg.M3A.MODE
        joint_type = cfg.M3A.MODAL_JOINT_TYPE

        dim_in = layer_dims[0]
        dim_inner = layer_dims[1]
        dim_out = layer_dims[2]

        self.relation_learner = nn.ModuleDict()
        if m3a_mode == "GRAPH":
            for i in range(nlayers):
                layer_dim_in = (dim_in if i == 0 else dim_inner)
                layer_dim_out = (dim_out if i == nlayers - 1 else dim_inner)
                self.relation_learner["graph_{}".format(i+1)] = GraphConvolution(layer_dim_in, layer_dim_out)
                if i < nlayers - 1:
                    self.relation_learner["graph_{}_relu".format(i+1)] = nn.LeakyReLU(0.2)

        elif m3a_mode == "TRSFMR":
            simplified = cfg.M3A.TRSFMR_SIMPLIFIED
            nhead = cfg.M3A.TRSFMR_NHEAD

            use_pos_encode = cfg.M3A.TRSFMR_POS_ENCODE
            if use_pos_encode:
                self.relation_learner["trsfmr_pos_encoder"] = PositionalEncoding(dim_in, dropout=0.2)

            for i in range(nlayers):
                if simplified:
                    self.relation_learner["trsfmr_encoder_{}".format(i+1)] = TransformerEncoderSimplified(dim_in, nhead, dropout=0.2)
                else:
                    self.relation_learner["trsfmr_encoder_{}".format(i+1)] = nn.TransformerEncoderLayer(dim_in, nhead, dim_inner, dropout=0.2)
                
                if i < nlayers - 1:
                    self.relation_learner["trsfmr_encoder_{}_relu".format(i+1)] = nn.LeakyReLU(0.2)
                
                if i == nlayers - 1 and joint_type != "CROSS":
                    self.relation_learner["trsfmr_encoder_{}_relu".format(i+1)] = nn.LeakyReLU(0.2)
                    self.relation_learner["trsfmr_decoder_{}_fc".format(i+1)] = nn.Linear(dim_in, dim_out)
                    self.relation_learner["trsfmr_decoder_{}_dropout".format(i+1)] = nn.Dropout(0.2)
                    self.init_weights(self.relation_learner["trsfmr_decoder_{}_fc".format(i+1)])

    def init_weights(self, fc_layer):
        initrange = 0.1
        nn.init.zeros_(fc_layer.weight)
        nn.init.uniform_(fc_layer.weight, -initrange, initrange)

    def forward(self, x, graph_adj=None):
        x_out = x

        for k, v in self.relation_learner.items():
            if "graph" in k and "relu" not in k:
                x_out = v(x_out, graph_adj)
            else:
                x_out = v(x_out)

        return x_out

class M3ARelationCrossFusion(nn.Module):
    def __init__(self, cfg, x_dims):
        super(M3ARelationCrossFusion, self).__init__()
        
        nhead = cfg.M3A.TRSFMR_NHEAD

        dim_in = x_dims[0]
        kdim = x_dims[1]
        vdim = x_dims[1] if len(x_dims) == 3 else x_dims[2]
        dim_out = x_dims[-1]
        
        self.relation_x_fusion = nn.ModuleDict()

        self.relation_x_fusion["trsfmr_decoder_cross"] = TransformerDecoderSimplified(dim_in, nhead, dropout=0.2, kdim=kdim, vdim=vdim)
        self.relation_x_fusion["trsfmr_decoder_cross_relu"] = nn.LeakyReLU(0.2)
        self.relation_x_fusion["trsfmr_decoder_cross_fc"] = nn.Linear(dim_in, dim_out)
        self.relation_x_fusion["trsfmr_decoder_cross_dropout"] = nn.Dropout(0.2)
        self.init_weights(self.relation_x_fusion["trsfmr_decoder_cross_fc"])
    
    def init_weights(self, fc_layer):
        initrange = 0.1
        nn.init.zeros_(fc_layer.weight)
        nn.init.uniform_(fc_layer.weight, -initrange, initrange)

    def forward(self, x_q, x_k, x_v):
        x_out = x_q

        for k, v in self.relation_x_fusion.items():
            if k == "trsfmr_decoder_cross":
                x_out = v(x_out, x_k, x_v)
            else:
                x_out = v(x_out)

        return x_out

class M3AJointLearning(nn.Module):
    def __init__(self, cfg, feature_size):
        super(M3AJointLearning, self).__init__()

        num_classes = cfg.MODEL.NUM_CLASSES
        mmit_version = cfg.DATA.MMIT_VERSION
        
        self.cfg = cfg
        self.mode = cfg.M3A.MODE
        self.multi_modal = nn.ModuleDict()
        
        if self.mode == "GRAPH":
            graph_adj_t = 0.4
            graph_adj_file = os.path.join(
                cfg.DATA.PATH_TO_DATA_DIR, "mmit-m3a", "graph-adj-{}.pkl".format(mmit_version))
            self.graph_adj = get_graph_adj_matrix(num_classes, graph_adj_t, graph_adj_file).float().cuda()
        
        if "AUDIO" in cfg.M3A.MODAL_TYPE:
            lexicon_file = os.path.join(
                cfg.DATA.PATH_TO_DATA_DIR, "mmit-m3a", "audio-lexicon-{}.pkl".format(mmit_version))
            self.audio_lexicon = get_audio_lexicon(lexicon_file)
        
        if "TEXT" in cfg.M3A.MODAL_TYPE:
            lexicon_file = os.path.join(
                cfg.DATA.PATH_TO_DATA_DIR, "mmit-m3a", "text-lexicon-{}.pkl".format(mmit_version))
            self.text_lexicon = get_text_lexicon(lexicon_file).float().cuda()
        
        layer_dim_param = {}
        layer_dim_param["RGB"] = (feature_size, cfg.M3A.HIDDEN_LAYER, num_classes)
        layer_dim_param["AUDIO"] = (cfg.M3A.AUDIO_VEC_SIZE, cfg.M3A.HIDDEN_LAYER, feature_size)
        layer_dim_param["TEXT"] = (cfg.M3A.TEXT_VEC_SIZE, cfg.M3A.HIDDEN_LAYER, feature_size)
        layer_dim_param["AUDIOTEXT"] = (cfg.M3A.AUDIO_VEC_SIZE + cfg.M3A.TEXT_VEC_SIZE,
                                     cfg.M3A.HIDDEN_LAYER, feature_size)
        
        m3a_nlayers = cfg.M3A.NLAYERS
        self.modal_joint_type = cfg.M3A.MODAL_JOINT_TYPE
        self.modal_list = cfg.M3A.MODAL_TYPE.split("_")
        for modal_type in self.modal_list:
            self.multi_modal[modal_type] = M3ARelationLearner(cfg, m3a_nlayers, layer_dim_param[modal_type])
        
        if self.modal_joint_type == "CROSS":
            cross_dim_list = []
            for modal_type in self.modal_list:
                cross_dim_list.append(layer_dim_param[modal_type][0])
            cross_dim_list.append(feature_size)
            self.multi_modal["CROSS_FUSION"] = M3ARelationCrossFusion(cfg, cross_dim_list)

    def run_relation_learner(self, x_input, modal_type):
        synonym_size = self.cfg.M3A.AUDIO_SYNONYM_SIZE
        rng_seed = self.cfg.RNG_SEED
        graph_adj=None
        if self.mode == "GRAPH":
            graph_adj = get_graph_adj_input(self.graph_adj).detach()
        
        if modal_type == "RGB":
            x_rgb = torch.unsqueeze(x_input, 1)
            x_rgb = x_rgb.expand(x_rgb.shape[0], self.cfg.MODEL.NUM_CLASSES, x_rgb.shape[2])
            x_rgb = self.multi_modal[modal_type](x_rgb, graph_adj)
            if self.modal_joint_type == "CROSS":
                return x_rgb
            else:
                x_out = x_rgb.mean(1)
        elif modal_type == "AUDIO":
            x_audio = get_audio_embedding(self.audio_lexicon, synonym_size, rng_seed)
            if self.mode == "GRAPH":
                x_audio = x_audio.squeeze()
            x_audio = self.multi_modal[modal_type](x_audio, graph_adj)
            if self.modal_joint_type == "CROSS":
                if "RGB" in self.modal_list and x_audio.size(0) != x_input.size(0):
                    x_audio = x_audio.expand(x_input.size(0), -1, -1)
                return x_audio
            else:
                x_audio = x_audio.squeeze().transpose(0, 1)
                x_out = torch.matmul(x_input, x_audio)
        elif modal_type == "TEXT":
            x_text = self.text_lexicon
            if self.mode != "GRAPH":
                x_text = x_text.unsqueeze(0)
            x_text = self.multi_modal[modal_type](x_text, graph_adj)
            if self.modal_joint_type == "CROSS":
                if "RGB" in self.modal_list and x_text.size(0) != x_input.size(0):
                    x_text = x_text.expand(x_input.size(0), -1, -1)
                return x_text
            else:
                x_text = x_text.squeeze().transpose(0, 1)
                x_out = torch.matmul(x_input, x_text)
        elif modal_type == "AUDIOTEXT":
            x_audio = get_audio_embedding(self.audio_lexicon, synonym_size, rng_seed)
            if self.mode == "GRAPH":
                x_audio = x_audio.squeeze()

            x_text = self.text_lexicon
            if self.mode != "GRAPH":
                x_text = x_text.unsqueeze(0)
            
            if self.mode == "GRAPH":
                x_audio_text = torch.cat((x_audio, x_text), 1).cuda()
            else:
                x_audio_text = torch.cat((x_audio, x_text), 2).cuda()
            x_audio_text = self.multi_modal[modal_type](x_audio_text, graph_adj)
            if self.modal_joint_type == "CROSS":
                if "RGB" in self.modal_list and x_audio_text.size(0) != x_input.size(0):
                    x_audio_text = x_audio_text.expand(x_input.size(0), -1, -1)
                return x_audio_text
            else:
                x_audio_text = x_audio_text.squeeze().transpose(0, 1)
                x_out = torch.matmul(x_input, x_audio_text)
        return x_out

    def forward(self, x):
        x_out_list = []

        for modal_type in self.modal_list:
            x_out_list.append(self.run_relation_learner(x, modal_type))

        if self.modal_joint_type != "CROSS":
            x_out = sum(x_out_list)
        else:
            num_modal = len(self.modal_list)
            x_cross = self.multi_modal["CROSS_FUSION"](x_out_list[0], x_out_list[1], 
                                                       x_out_list[1] if num_modal == 2 else x_out_list[2])
            if "RGB" in self.modal_list:
                x = x.unsqueeze(1)
                x_cross = x_cross.transpose(1, 2)
                x_out = torch.matmul(x, x_cross).squeeze()
            else:
                x_cross = x_cross.squeeze().transpose(0, 1)
                x_out = torch.matmul(x, x_cross)
        return x_out

class M3AHead(nn.Module):
    def __init__(self, cfg, feature_size, pool_size):
        super(M3AHead, self).__init__()
        
        self.cfg = cfg
        self.avgpool = nn.AvgPool3d(pool_size, stride=(1, 1, 1), padding=(0, 0, 0))
        
        self.m3a_joint_learning = M3AJointLearning(cfg=cfg, feature_size=feature_size)

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

        x = self.m3a_joint_learning(x)

        if not self.training:
            x = self.act(x)
        
        return x
