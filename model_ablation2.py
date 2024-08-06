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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ReductionBlock(nn.Module):
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
        
        self.reduct = ReductionBlock(dim, hidden_dim)

        for p in self.embedding_layer.parameters():
          p.requires_grad = False

        

    def forward(self, x):
        
        x = self.embedding_layer(x)

        x = x.to(torch.float32)
        
        out = self.reduct(x)

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


class MQParallelFFTransformer(nn.Module):
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

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.norm_out = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_fused, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


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

        x = self.norm1(x)
        y = self.norm2(y)

        
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

        out = self.norm_out(out + y)

        out = out + self.feedforward(y)

        return out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, 
                 dim,
                 dim_head=32, 
                 heads=8, 
                 ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
       
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)


        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(dim * heads, dim, bias=False)


        self.feedforward1 = nn.Linear(dim, dim * 2, bias=False)
        self.feedforward2 = nn.Linear(dim * 2, dim, bias=False)

        self.activation = F.relu

       

    def forward(self, x, y):
        """ 
        x fused into y
        """
        
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        attn = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = attn * self.scale

        attn = nn.Softmax(dim= -1)(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        out = y + self.norm1(out)

        out1 = self.feedforward2(self.activation(self.feedforward1(out)))

        out = out1 + out

        out = self.norm2(out)

        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, 
                 dim,
                 dim_head=32, 
                 heads=8, 
                 ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
       
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)


        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(dim * heads, dim, bias=False)


        self.feedforward1 = nn.Linear(dim, dim * 2, bias=False)
        self.feedforward2 = nn.Linear(dim * 2, dim, bias=False)

        self.activation = F.relu

       

    def forward(self, x):
        """ 
        x fused into y
        """
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        attn = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = attn * self.scale

        attn = nn.Softmax(dim= -1)(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out) + x

        out = self.norm1(out)

        out = self.feedforward2(self.activation(self.feedforward1(out)))

        out = self.norm2(out)

        return out


    
    

class FuseAttentionBlock(nn.Module):
    def __init__(self, dim, dim_fused):
        super().__init__()
        self.W1 = nn.Linear(dim, dim_fused, bias=False)
        
        self.W2 = nn.Linear(dim_fused * 2, 1)
        
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)
                  
                 
                 
    def forward(self, x):
        '''
        x: b n d
        '''

        f = self.W1(x) # b n dim_fused


        f1, f2 = f[0], f[1]


        a12 = self.W2(self.act(torch.cat([f1, f2], dim=-1)))
        
        
        a21 = self.W2(self.act(torch.cat([f2, f1], dim=-1)))
    
        
        coef = torch.cat([a12, a21])

        coef = F.softmax(coef, dim=-1)

        return coef
 

class PoiEnhancer(nn.Module):
    def __init__(self,  llm_e_path_a, llm_e_path_c, llm_e_path_t, poi_e_path, cross_layer_num=3, dim=256):
        
        super().__init__()
        
        self.llm_layer_a = EmbeddingBlock(embed_path = llm_e_path_a, hidden_dim=dim)
        self.llm_layer_c = EmbeddingBlock(embed_path = llm_e_path_c, hidden_dim=dim)
        self.llm_layer_t = EmbeddingBlock(embed_path = llm_e_path_t, hidden_dim=dim)

    
        self.poi_layer = EmbeddingBlock(embed_path = poi_e_path, hidden_dim = dim, dim_reduct=False)

    
        self.llm_e_dim = self.llm_layer_a.get_shape()[1]
        self.poi_e_dim = self.poi_layer.get_shape()[1]


        self.cross_layer_num = cross_layer_num
        
        self.nearby_alignment = nn.ModuleList([])
        for _ in range(cross_layer_num):
            self.nearby_alignment.append(
                CrossAttentionTransformer(dim=self.poi_e_dim)
            )

        self.time_alignment = nn.ModuleList([])
        for _ in range(cross_layer_num):
            self.time_alignment.append(
                CrossAttentionTransformer(dim=self.poi_e_dim)
            )

        self.semantic_attention_fusion =  FuseAttentionBlock(dim=self.poi_e_dim, dim_fused= 2 * self.poi_e_dim)

        self.dual_modal_fuse_norm = LayerNorm(dim)
        self.dual_modal_fusion = nn.ModuleList([])
        for _ in range(cross_layer_num):
            self.dual_modal_fusion.append(
                Residual(MQParallelFFTransformer(dim=self.poi_e_dim, dim_fused=self.poi_e_dim))
            )

        self.to_embedding = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim, bias=False)
        )

        self.apply(weight_init)


    def forward(self, batch):
    
        llm_e_a_1 = self.llm_layer_a(batch)
        llm_e_a_2 = llm_e_a_1
        
        llm_e_c = self.llm_layer_c(batch)
        llm_e_t = self.llm_layer_t(batch)

        
    
        for block_nearby in self.nearby_alignment:
            llm_e_a_1 = block_nearby(llm_e_c, llm_e_a_1)
        
        for block_time in self.time_alignment:
            llm_e_a_2 = block_time(llm_e_t, llm_e_a_2)

        out = torch.stack([llm_e_a_1, llm_e_a_2])

        out1 = rearrange(out, 'fn b n d -> fn (b n) d')
        
        coef = self.semantic_attention_fusion(out1)

        temp_out = coef[0] * out[0] + coef[1] * out[1] 

        y = self.poi_layer(batch)
        y_ = y

        for block in self.dual_modal_fusion:
            y = block(temp_out, y)

        z = self.to_embedding(y)

        
        return z, y_






    


