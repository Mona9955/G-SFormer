import math
import logging
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


joints_left = [4, 5, 6, 11, 12, 13]
joints_right = [1, 2, 3, 14, 15, 16]

left_arm_keys = [11,12,13]
right_arm_keys = [14,15,16]
left_leg_keys = [4,5,6]
right_leg_keys = [1,2,3]
trunk_keys = [0,7,8,9,10]

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#pose_seq: bs, T, J, 2
class Sparse_Graph(nn.Module):
    def __init__(self, dim_p=16, dim_p1=51, dim_k=30, out_dim = 256, res=True):
        super(Sparse_Graph, self).__init__()
        self.enco_limbs = nn.Linear(6, dim_p, bias=False)
        self.enco_limbs_1 = nn.Linear(dim_p, dim_p1)
        self.enco_trunk = nn.Linear(10, dim_p, bias=False)
        self.enco_trunk_1 = nn.Linear(dim_p, dim_p1)

        self.enco_keys = nn.Linear(2, dim_k, bias=False)
        self.parts_embed = nn.Parameter(torch.randn(1, 5, dim_p1))
        

        self.conv_graph = nn.Conv2d(dim_p1, 1, (1,1))
        self.res = res
        self.dim_p1 = dim_p1
        self.layer_norm =LayerNorm(dim_k*17)
        self.linear_out = nn.Linear(dim_k*17, out_dim, bias=False)
        self.re1 = nn.ReLU()
        self.re2 = nn.Tanh()

    def enco_parts(self, pose_seq):
        left_arm_enco = self.re1(self.enco_limbs(torch.flatten(pose_seq[:,:, left_arm_keys, :], start_dim=2, end_dim=3)))
        left_arm_enco = self.re2(self.enco_limbs_1(left_arm_enco))
        left_leg_enco = self.re1(self.enco_limbs(torch.flatten(pose_seq[:,:, left_leg_keys, :], start_dim=2, end_dim=3)))
        left_leg_enco = self.re2(self.enco_limbs_1(left_leg_enco))
        right_arm_enco = self.re1(self.enco_limbs(torch.flatten(pose_seq[:,:, right_arm_keys, :], start_dim=2, end_dim=3)))
        right_arm_enco = self.re2(self.enco_limbs_1(right_arm_enco))
        right_leg_enco = self.re1(self.enco_limbs(torch.flatten(pose_seq[:,:, right_leg_keys, :], start_dim=2, end_dim=3)))
        right_leg_enco = self.re2(self.enco_limbs_1(right_leg_enco))

        trunk_enco = self.re1(self.enco_trunk(torch.flatten(pose_seq[:,:, trunk_keys, :], start_dim=2, end_dim=3)))
        trunk_enco = self.re2(self.enco_trunk_1(trunk_enco))

        parts_enco = torch.stack((left_arm_enco, left_leg_enco, right_arm_enco, right_leg_enco, trunk_enco), dim=2)  #bs, T, 5, 51

        if self.res:
            keys_enco = self.enco_keys(pose_seq)   # bs, T, 17, 30
            #keys_enco += self.pos_embed
            return parts_enco, keys_enco

        return parts_enco


    def compute_edge(self, parts_enco_s):
        # parts_enco_s = parts_enco.view(-1, 5, self.dim_p1)
        A1 = parts_enco_s[:,:,None,:] + parts_enco_s[:,None,:,:]
        A1 = A1.permute(0,3,1,2).contiguous()
        A1 = self.conv_graph(A1).squeeze(dim=1)
        A1 = torch.sigmoid(A1).masked_fill(torch.eye(5).bool().cuda(), 1)
        return A1

    def forward(self, input_2d_seq):
        bs, T, J, _ = input_2d_seq.shape
        if self.res:
            parts_enco, keys_enco = self.enco_parts(input_2d_seq)

        else:
            parts_enco = self.enco_parts(input_2d_seq)
        parts_enco_s = parts_enco.view(-1, 5, self.dim_p1)   #bs*T, 5, 51
        parts_enco_s += self.parts_embed
        adj_mat = self.compute_edge(parts_enco_s)  #bs*T, 5, 5
        parts_graph = self.re1(torch.matmul(adj_mat, parts_enco_s) - parts_enco_s)   #bs*T, 5, 51
        parts_s = torch.cat((parts_graph, parts_enco_s), dim=-1).view(bs, T, -1)    # 51*5*2 = 510
        if self.res:
            parts_s = parts_s + keys_enco.flatten(start_dim=2, end_dim=3)  #bs, T, 510

        parts_s = self.layer_norm(parts_s)
        parts_s = self.linear_out(parts_s)

        return parts_s

