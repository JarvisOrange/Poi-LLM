import torch
from torch import einsum, nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pandas as pd

from einops import rearrange, repeat

from torch.utils import data
from torch.utils.data import DataLoader	
from info_nce import InfoNCE, info_nce

from tqdm import *

from poi_utils import *

import os


class Embed2hidden(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.to_hidden = nn.Linear(dim, hidden_dim, bias=False).float()

    def forward(self, x):
        hidden = self.to_hidden(x)
        return F.normalize(hidden, dim=-1)      

class EmbeddingBlock(nn.Module):
    def __init__(self, hidden_dim=256,  dim_reduct=True, embed_path=''):
        super().__init__()
        self.embed = torch.load(embed_path)
        num, dim = self.embed.shape
        self.shape = self.embed.shape

        self.embedding_layer = nn.Embedding(num, dim, _weight=self.embed)
        
        self.dim_reduct = dim_reduct
        if self.dim_reduct == True: 
            self.embed2hidden = Embed2hidden(dim, hidden_dim)
        else:
            self.embed2hidden = nn.Identity()

        for p in self.embedding_layer.parameters():
          p.requires_grad = False

        

    def forward(self, x):
        
        x = self.embedding_layer(x)

        x = x.to(torch.float32)
        
        out = self.embed2hidden(x)

        return out

    def get_shape(self):
        return self.shape


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class CrossAttentionBlock(nn.Module):
    def __init__(self, 
                 dim,
                 dim_fused, 
                 dim_head=32, 
                 heads=8, 
                 ff_mult=4
                 ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        dim_fused = dim if dim_fused != None else 256

        self.norm = LayerNorm(dim)
        self.norm_ = LayerNorm(dim) if dim_fused else nn.Identity()

        self.norm_out = LayerNorm(dim_fused)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_fused, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.feedforward = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

       

    def forward(self, x, y):
        """ 
        b - batch
        d - feature dimension
        """

        x_ = x

        x = self.norm(x)
        y = self.norm(y)

        
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        k, v = self.to_kv(y).chunk(2, dim=-1)

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        out = out + self.feedforward(x)
        

        return out
    

    
class SelfAttentionBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 dim_head=32, 
                 heads=8, 
                 ff_mult=4
                 ):

        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        self.norm = LayerNorm(dim)

        self.norm_out = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.feedforward = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

    def forward(self, x):
        """ 
        b - batch
        d - feature dimension
        """

        x_ = x
        x = self.norm(x)
        

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
 

        q = q * self.scale

        k, v = self.to_kv(x).chunk(2, dim=-1)

  

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # out = out.squeeze(1) # b 1 d -> b  d
        out = self.to_out(out)


        out = out + self.feedforward(x)



        return out
    
    

class FuseAttentionBlock(nn.Module):
    def __init__(self, dim, dim_fused):
        super().__init__()
        self.W = nn.Linear(dim, dim_fused, bias=False)
        self.f = nn.Linear(dim_fused * 2, 1)
        
        # self.f1 = nn.Conv1d(dim_fused, 1, kernel_size=1)
        # self.f2 = nn.Conv1d(dim_fused, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)
                  
                 
                 
    def forward(self, x):
        '''
        x: b n d
        '''
        # seq_fts = self.W(src)
        # print(seq_fts.shape) #1536 1 8
        # f_1 = self.f1(seq_fts)
        # f_2 = self.f2(seq_fts)
        # print(f_1.shape) #1536 1 8
        # print(f_2.transpose(1, 2).shape) #1536 8 1
        # logits = f_1 + f_2.transpose(1, 2)
        # print(logits.shape) #1536 8 8
        # coefs = torch.mean(self.act(logits), dim=-1)
        # print(coefs.shape) #1536  8
        # coefs = torch.mean(coefs, dim=0)
        # print(coefs.shape) #1536  8
        # coefs = F.softmax(coefs, dim=-1)
 
        # exit()

        f = self.W(x) # b n dim_fused


        f1, f2, f3 = f[0], f[1], f[2]


        a12 = self.act(self.f(torch.cat([f1, f2], dim=-1)))
        a13 = self.act(self.f(torch.cat([f1, f3], dim=-1)))


        
        a21 = self.act(self.f(torch.cat([f2, f1], dim=-1)))
        a23 = self.act(self.f(torch.cat([f2, f3], dim=-1)))

        a31 = self.act(self.f(torch.cat([f3, f1], dim=-1)))
        a32 = self.act(self.f(torch.cat([f3, f2], dim=-1)))

        a1 = torch.mean(a12 + a13, dim=0)


        a2 = torch.mean(a21 + a23, dim=0)

        a3 = torch.mean(a31 + a32, dim=0)
        
        coef = torch.cat([a1,a2,a3])

        coef = F.softmax(coef, dim=-1)

        return coef
 

class PoiEnhancer(nn.Module):
    def __init__(self,  llm_e_path1, llm_e_path2, llm_e_path3, poi_e_path, cross_layer_num=1, dim=256):
        
        super().__init__()
        
        self.llm_layer1 = EmbeddingBlock(embed_path = llm_e_path1)
        self.llm_layer2 = EmbeddingBlock(embed_path = llm_e_path2)
        self.llm_layer3 = EmbeddingBlock(embed_path = llm_e_path3)

        self.poi_layer = EmbeddingBlock(embed_path = poi_e_path, hidden_dim = dim, dim_reduct=False)

        self.llm_e_dim = self.llm_layer1.get_shape()[1]
        self.poi_e_dim = self.poi_layer.get_shape()[1]



        self.attention_block1 = SelfAttentionBlock(dim=self.poi_e_dim)
        self.attention_block2 = SelfAttentionBlock(dim=self.poi_e_dim)
        self.attention_block3 = SelfAttentionBlock(dim=self.poi_e_dim)

        self.fuse_attention =  FuseAttentionBlock(dim=self.poi_e_dim, dim_fused= 2 * self.poi_e_dim)

        self.cross_layer_num = cross_layer_num

        self.cross_attention_fusion = nn.ModuleList([])
        for ind in range(cross_layer_num):
            self.cross_attention_fusion.append(
                CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim)
            )

        self.apply(weight_init)

        

        

    def forward(self, batch):
    
        llm_e1 = self.llm_layer1(batch)
        
        llm_e2 = self.llm_layer2(batch)
        llm_e3 = self.llm_layer3(batch)

        y = self.poi_layer(batch)

        x1 = self.attention_block1(llm_e1)
        x2 = self.attention_block2(llm_e2)
        x3 = self.attention_block3(llm_e3)

        out = torch.stack([x1, x2, x3])

        out1 = rearrange(out, 'fn b n d -> fn (b n) d')
        
        coef = self.fuse_attention(out1)


        temp_out = coef[0] * out[0] + coef[1] * out[1] + coef[2] * out[2]

        for cross_attention_fusion in self.cross_attention_fusion:
            temp_out = cross_attention_fusion(temp_out, y)


        z = temp_out

        
        return z, y






    


