import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model_spatial import Sparse_Graph
import numpy as np
import math
import os
from timm.models.layers import DropPath, trunc_normal_
import copy

from common.arguments import parse_args

opt = parse_args()
seq_len = opt.number_of_frames

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = layers
        # self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, layer, d_model_ls):
        super(Decoder, self).__init__()
        self.layers = layer
            # clones(layer, N)
        # self.norm = LayerNorm(layer.size)
        self.pos_embedding_1 = nn.Parameter(torch.randn(1, seq_len, d_model_ls[0]))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, seq_len // 3, d_model_ls[1]))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, seq_len // 9, d_model_ls[2]))

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x += self.pos_embedding_1
            elif i == 1:
                x += self.pos_embedding_2
            elif i == 2:
                x += self.pos_embedding_3
            x = layer(x, mask)
        return x


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


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, size_in, size_out, dropout, stride=1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size_in)
        self.trans = None if size_in==size_out else nn.Linear(size_in, size_out)
        self.dropout = DropPath(dropout)
        # self.dropout = nn.Dropout(dropout)
        self.stride = stride
        self.pooling = nn.MaxPool1d(1, stride)
        # self.norm1 = LayerNorm(size_out)

    def forward(self, x, sublayer):
        res = x
        if self.stride != 1:
            res = self.pooling(x.permute(0,2,1).contiguous())
            res = res.permute(0,2,1)
            # return res + self.dropout(sublayer(self.norm(x)))
        if self.trans:
            res = self.trans(res)
            # res = self.norm1(res)

        return res + self.dropout(sublayer(self.norm(x)))



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.gelu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, size_in, size_out, dropout, stride=1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.Sequential(SublayerConnection(size_in, size_out, dropout, stride), SublayerConnection(size_out, size_out, dropout))
            # clones(SublayerConnection(size, dropout, stride), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):   #dropout=0
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout!=0 else None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# n, t, dim  -> n,t/3, dim
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.ReLU()

    def forward(self, input, nonl=True):
        input = input.permute(0,2,1)
        out = self.depth_conv(input)
        if nonl:
            out = self.dropout(self.gelu(out))
        out = self.point_conv(out)
        out = out.permute(0,2,1)
        return out

# input: bs, t, c
class Super_MHAT(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1, M=9, kernal=5, n_iter=1):
        super(Super_MHAT, self).__init__()
        self.m = M
        self.n_iter = n_iter
        self.kernal = kernal
        self.eps = 1e-12
        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout) if dropout != 0. else None


    def forward(self, x, mask=None):
        N, T, C = x.shape
        scale = C ** -0.5
        pad = (self.kernal-1)//2
        T_ = T // self.m
        tokens = x.view(N, T_, self.m, C)  #n, t/9, 9, c
        stokens = F.avg_pool1d(x.permute(0,2,1), self.m).unsqueeze(dim=-1)   #b, c, t/9, 1
        with torch.no_grad:
            for idx in range(self.n_iter):
                stokens = F.unfold(stokens, kernel_size=(self.kernal,1), padding=(pad, 0))  #b, c*5, t/9
                stokens = stokens.transpose(1,2).reshape(N, -1, C, self.kernal)   #n, t/9, c, 5
                association = tokens @ stokens * scale   #n, t/9, 9, 5
                association = association.softmax(-1)
                association_sum = association.sum(2).transpose(1,2).reshape(N, self.kernal, -1)  #n, 5, t/9
                association_sum = F.fold(association_sum, output_size=(T_, 1), kernel_size=(self.kernal, 1), padding=(pad, 0)).squeeze(dim=-1)   #n, 1, t/9

        stokens = tokens.transpose(-1,-2) @ association  #b, t/9, c, 5
        stokens = stokens.permute(0,3,2,1).reshape(N, -1, T_)  #b, c*5, t/9
        stokens = F.fold(stokens, output_size=(T_, 1), kernel_size=(self.kernal, 1), padding=(pad, 0))   #b, c, t/9, 1
        stokens = stokens.squeeze(dim=-1) /(association_sum + self.eps)

        stokens = stokens.permute(0,2,1)  #b, t/9, c
        query, key, value = \
            [l(x).view(N, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (stokens, stokens, stokens))]  # b, h, t/9, d_k
        stokens, atten_stoken = attention(query, key, value, mask=mask, dropout=self.dropout)
        stokens = stokens.transpose(1, 2).contiguous().view(N, -1, self.h * self.d_k)   #b, t/9, c
        stokens = self.linears[-1](stokens).permute(0,2,1).unsqueeze(dim=-1)  #b, c, t/9, 1

        stokens = F.unfold(stokens, kernel_size=(self.kernal, 1), padding=(pad, 0))
        stokens = stokens.transpose(1,2).reshape(N, T_, C, self.kernal)
        tokens = stokens @ association.transpose(-1,-2)   #n, t/9, c, 5     n, t/9, 5, 9  -> n, t/9, c, 9
        tokens = tokens.transpose(-1, -2).reshape(N, T, C)
        return tokens


# input: bs, t, c
class Sparse_MHAT(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1, M=3):
        super(Sparse_MHAT, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.h = h
        self.d_k = d_model // h
        self.m = M
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, pad=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches, len, input_size = x.shape
        sparse_output = torch.zeros_like(x).to('cuda')
        if len % self.m != 0:
            if pad:
                pad_num = self.m - len%self.m
                x = torch.cat((x, x[:,-2:-1,:].repeat(1, pad_num, 1)), dim=1)
            else:
                del_num = len % self.m
                x = x[:, :-del_num, :]

            len = x.shape[1]

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]   #nbatches, h, t, d_k

        for i in range(self.m):
            q_i = query[:,:,i:(len+1-self.m+i):self.m,:]
            k_i = key[:,:,i:(len+1-self.m+i):self.m,:]
            v_i = value[:,:,i:(len+1-self.m+i):self.m,:]
            x, self.attn = attention(q_i, k_i, v_i, mask=mask, dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            sparse_output[:,i:(len+1-self.m+i):self.m,:] = x

        sparse_output = self.linears[-1](sparse_output)

        return sparse_output


# input: bs, t, c
class Sparse_MHAT_reduce(nn.Module):
    def __init__(self, h, d_model, d_out, dropout=0.1, M=3):
        super(Sparse_MHAT_reduce, self).__init__()
        self.m = M
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(p=dropout) if dropout !=0. else None
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linear_1 = nn.Linear(M*d_model, d_out)
        self.act = nn.GELU()

    def forward(self, x, mask=None, pad=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches, len, input_size = x.shape
        sparse_output = torch.empty(size=(nbatches,0,input_size)).to('cuda')
        if len % self.m != 0:
            if pad:
                pad_num = self.m - len%self.m
                x = torch.cat((x, x[:,-2:-1,:].repeat(1, pad_num, 1)), dim=1)
            else:
                del_num = len % self.m
                x = x[:, :-del_num, :]

            len = x.shape[1]

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]   #nbatches, h, t, d_k

        for i in range(self.m):
            q_i = query[:,:,i:(len+1-self.m+i):self.m,:]
            k_i = key[:,:,i:(len+1-self.m+i):self.m,:]
            v_i = value[:,:,i:(len+1-self.m+i):self.m,:]
            x, self.attn = attention(q_i, k_i, v_i, mask=mask, dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            if i == 0:
                sparse_output = x
            else:
                sparse_output = torch.cat((sparse_output, x), dim=-1)

        sparse_output = self.linear_1(sparse_output)
        return sparse_output   # n, t, d_modl*3





# input = N, T, 2(256/512)
class Transformer(nn.Module):
    def __init__(self, deco_n = 5, d_model=256, d_ff=512, h=8, dropout=0., drop_path_rate=0.1, length=81, depth_wise=False):
        super(Transformer, self).__init__()
        self.spatial_enco = Sparse_Graph(out_dim=d_model)
        self.depth_wise = depth_wise
        self.num_joints_out = 17
        self.length = length
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, deco_n)]
        if depth_wise:
            self.depth_wise = DepthWiseConv(d_model, d_model)
        else:

            self.pos_embedding = nn.Parameter(torch.randn(1, length, d_model))

        self.model_enco = self.make_model_enco(d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        self.model_deco = self.make_model_deco(d_model=d_model, d_ff=d_ff, N=deco_n, h=h, dropout=dropout)
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(d_model, momentum=0.1),
            nn.Conv1d(d_model, 3 * self.num_joints_out, kernel_size=1)
            # nn.Linear(d_model, 3 * self.num_joints_out)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(d_model, momentum=0.1),
            nn.Conv1d(d_model, 3 * self.num_joints_out, kernel_size=1)
        )


    def forward(self, x, mask=None):
        N,T,J,_ = x.shape
        x = self.spatial_enco(x) #N,T,256
        if self.depth_wise:
            pos_embedding = self.depth_wise(x)  #self.pos_embedding
        else:
            pos_embedding = self.pos_embedding
        x += pos_embedding
        x = self.model_enco(x, mask)
        enco_out = x.permute(0,2,1).contiguous()
        enco_out = self.fcn(enco_out)
        enco_out = enco_out.permute(0,2,1).contiguous().view(N,T,J,-1)#n, t, j*3
        deco_out = self.model_deco(x, mask)
        deco_out = deco_out.permute(0,2,1).contiguous()
        deco_out = self.fcn_1(deco_out)  #n, j*3, 1
        deco_out = deco_out.permute(0,2,1).contiguous().view(N,1,J,-1)

        return deco_out, enco_out

    def make_model_enco(self, d_model=256, d_ff=512, h=8, dropout=0.1):
        c = copy.deepcopy
        # attn1 = Super_MHAT(h, d_model, M=9, dropout=dropout)
        # attn2 = Super_MHAT(h, d_model, M=3, dropout=dropout)
        attn1 = Sparse_MHAT(h, d_model, M=3, dropout=dropout)
        attn2 = Sparse_MHAT(h, d_model, M=3, dropout=dropout)
        attn3 = Sparse_MHAT(h, d_model, M=3, dropout=dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, d_model, dropout)
        enco_layers = nn.ModuleList([EncoderLayer(attn1, c(ff), d_model, d_model, dropout),
                                     EncoderLayer(attn2, c(ff), d_model, d_model, dropout),
                                     EncoderLayer(attn3, c(ff), d_model, d_model, dropout)])
        # model = Encoder(EncoderLayer(c(attn), c(ff), d_model, d_model, dropout), N)
        model = Encoder(enco_layers)
        return model

    def make_model_deco(self,  d_model=256, d_ff=512, N=5, h=8, dropout=0.1):
        embed_dims = [d_model, d_model, d_model]
        layers = []
        for i in range(N):
            attn = Sparse_MHAT_reduce(h, d_model, dropout=dropout, d_out=d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, d_model, dropout)
            layers.append(EncoderLayer(attn, ff, d_model, d_model, dropout, stride=3))
            # DepthWiseConv(d_model, d_model)

        enco_layers = nn.ModuleList(layers)
        model = Decoder(enco_layers, embed_dims)
        return model

if __name__ == '__main__':
    inputs_2d = torch.randn(size=(2, 243, 17, 2)).cuda()
    model = Transformer(length=243, dropout=0.).cuda()
    deco_out, enco_out = model(inputs_2d)
    print(deco_out.shape)
    print(enco_out.shape)

