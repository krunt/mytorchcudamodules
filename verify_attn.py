import torch
import torch.nn.functional as F
from torch import nn
import argparse
import numpy as np
import random

import modules.initialize as minit

minit.initialize(verbose=False)

from modules import SelfMultiheadAttn

import math

class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3*embed_dim)  # don't change the name
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # don't change the name

        self.norm_coeff = self.head_dim ** 0.5

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, qkv):
        """
        qkv - query, key and value - it should be the same tensor since we implement self-attention
        """
        # YOUR CODE
        # 1. apply self.in_proj to qkv
        # 2. split the result of step 1 on three equal parts of size self.embed_dim: query, key, value
        # 3. compute scaled dot-product attention for different heads in loop.
        #    i-th head will work on query[:, :, i*head_dim: (i+1)*head_dim],
        #    key[:, :, i*head_dim: (i+1)*head_dim], value[:, :, i*head_dim: (i+1)*head_dim]
        # 4. apply dropout for each head result
        # 5. concat all results
        # 6. apply self.out_proj to the result of step 5

        embed_dim = self.embed_dim
        head_dim = self.head_dim

        coef_inv = 1 / self.norm_coeff

        qkv3 = self.in_proj(qkv)

        #print('q', qkv3)

        q, k, v = qkv3[:, :, :embed_dim], qkv3[:, :, embed_dim:2*embed_dim], qkv3[:, :, 2*embed_dim:]

        #buf = torch.zeros_like(q)
        lst = []
        for i in range(self.num_heads):
            qi = q[:, :, i*head_dim: (i+1)*head_dim]
            ki = k[:, :, i*head_dim: (i+1)*head_dim]
            vi = v[:, :, i*head_dim: (i+1)*head_dim]

            tmp = torch.matmul(qi, ki.transpose(1,2)) * coef_inv
            tmp = F.softmax(tmp, dim=-1)
            tmp = torch.matmul(tmp, vi)

            #buf[:, :, i*head_dim: (i+1)*head_dim] = tmp

            lst.append(tmp)

        buf = torch.cat(lst, dim=-1)
        buf = self.attention_dropout(buf)

        assert buf.shape[-1] == embed_dim

        result = self.out_proj(buf)

        #raise NotImplementedError
        return result

#seq_length=1024
#num_seqs=10
#hidden_dim=1024
#heads=16

#seq_length=2
#num_seqs=2
#hidden_dim=2
#heads=2

seq_length=77
num_seqs=2
hidden_dim=1024
heads=16

gtruth_mha = torch.nn.MultiheadAttention(hidden_dim, heads, bias=True, dropout=0, batch_first=True)

test_mha = SelfMultiheadAttn(hidden_dim, heads, bias=True, dropout=0, include_norm_add=False, impl='default')
test_mha.in_proj_weight = gtruth_mha.in_proj_weight
test_mha.in_proj_bias = gtruth_mha.in_proj_bias
test_mha.out_proj_weight = gtruth_mha.out_proj.weight
test_mha.out_proj_bias = gtruth_mha.out_proj.bias

test_mha2 = CustomMultiHeadSelfAttention(embed_dim=hidden_dim, num_heads=heads, dropout=0)
test_mha2.in_proj.weight = gtruth_mha.in_proj_weight
test_mha2.in_proj.weight
test_mha2.in_proj.bias = gtruth_mha.in_proj_bias
test_mha2.out_proj.weight = gtruth_mha.out_proj.weight
test_mha2.out_proj.weight
test_mha2.out_proj.bias = gtruth_mha.out_proj.bias

device = torch.device('cuda')
gtruth_mha.to(device)
test_mha.to(device)
test_mha2.to(device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

for _ in range(1):
    a = torch.rand((num_seqs, seq_length, hidden_dim), device=device)
    #a = torch.rand((num_seqs, seq_length, hidden_dim), device=device)
    out0 = gtruth_mha(a, a, a)[0].cpu().detach().numpy()
    out1 = test_mha(a, a, a)[0].cpu().detach().numpy()
    #out2 = test_mha2(a.transpose(0, 1)).transpose(0, 1).cpu().detach().numpy()
    #out2 = test_mha2(a.transpose(0,1)).transpose(0,1).cpu().detach().numpy()
    out2 = test_mha2(a).cpu().detach().numpy()
    assert np.allclose(out0, out2, atol=1e-4), f"{out0} {out2}"
    assert np.allclose(out0, out1, atol=1e-4), f"{out0} {out1}"

print ("Congratulations! It works!")

