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


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FusedAttentionBlock(nn.Module):
    def __init__(self, dim, dim_fused):
        super().__init__()
        self.fuse1 = CrossAttentionBlock(dim, dim_fused,  heads=1) 
        self.fuse2 = CrossAttentionBlock(dim, dim_fused,  heads=1) 
        self.fuse3 = CrossAttentionBlock(dim, dim_fused,  heads=1) 
        self.fuse4 = CrossAttentionBlock(dim, dim_fused,  heads=1) 
        self.fuse5 = CrossAttentionBlock(dim, dim_fused,  heads=1) 
        self.fuse6 = CrossAttentionBlock(dim, dim_fused,  heads=1) 
                  
                 
                 
    def forward(self, x1, x2, x3, x4, x5, x6, y):
        """ 
        b - batch
        d - feature dimension
        """

        z1 =  self.fuse1(x1, y)
        z2 =  self.fuse1(x2, y)
        z3 =  self.fuse1(x3, y)
        z4 =  self.fuse1(x4, y)
        z5 =  self.fuse1(x5, y)
        z6 =  self.fuse1(x6, y)
        

        return torch.cat([z1,z2,z3,z4,z5,z6], dim = -1)



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
        self.norm_ = LayerNorm(dim_fused) if dim_fused else nn.Identity()

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
        x = self.norm(x)
        y = self.norm_(y)
        

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

        out = out.squeeze(1) # b 1 d -> b  d
        out = self.to_out(out)

        out = out + self.feedforward(x)

        return out
        

class Embed2hidden(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.to_hidden = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        hidden = self.to_hidden(x)
        return F.normalize(hidden, dim=-1)       
        

class EmbeddingBlock(nn.Module):
    def __init__(self, hidden_dim=256,  dim_reduct=True, embed_path=''):
        super().__init__()
        self.embed = torch.load(embed_path)
        num, dim = self.embed.shape
        self.shape = self.embed.shape

        self.embedding_layer = nn.Embedding(num, dim, _weight=self.embed).half()
        
        self.dim_reduct = dim_reduct
        if self.dim_reduct: 
            self.embed2hidden = Embed2hidden(dim, hidden_dim)
        else:
            self.embed2hidden = nn.Identity()
        

    def forward(self, x):
        with torch.no_grad():
            x = self.embedding_layer(x)
        out = self.embed2hidden(x)
        return out

    def get_shape(self):
        return self.shape
 

class PoiEnhancer(nn.Module):
    def __init__(self,  llm_e_path1, llm_e_path2, llm_e_path3, poi_e_path):
        
        super().__init__()
        
        self.llm_layer1 = EmbeddingBlock(embed_path = llm_e_path1)
        self.llm_layer2 = EmbeddingBlock(embed_path = llm_e_path2)
        self.llm_layer3 = EmbeddingBlock(embed_path = llm_e_path3)

        self.poi_layer = EmbeddingBlock(embed_path = poi_e_path, dim_reduct=False)

        self.llm_e_dim = self.llm_layer1.get_shape()[1]
        self.poi_e_dim = self.poi_layer.get_shape()[1]

        self.attention_block1 = SelfAttentionBlock(dim=self.poi_e_dim)
        self.attention_block2 = SelfAttentionBlock(dim=self.poi_e_dim)
        self.attention_block3 = SelfAttentionBlock(dim=self.poi_e_dim)

        self.cross_attention_block12 = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim) 
        self.cross_attention_block13 = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim) 
        self.cross_attention_block21 = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim) 
        self.cross_attention_block23 = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim) 
        self.cross_attention_block31 = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim) 
        self.cross_attention_block32 = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim) 

        self.fused_attention_block = FusedAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim)


        

    def forward(self, batch):
    
        llm_e1 = self.llm_layer1(batch)
        
        llm_e2 = self.llm_layer2(batch)
        llm_e3 = self.llm_layer3(batch)

        y = self.poi_layer(batch)

        
        x1 = self.attention_block1(llm_e1)
        x2 = self.attention_block2(llm_e2)
        x3 = self.attention_block3(llm_e3)

        

        x12 = self.cross_attention_block12(x1, x2)
        x13 = self.cross_attention_block13(x1, x2)
        x21 = self.cross_attention_block21(x1, x2)
        x23 = self.cross_attention_block23(x1, x2)
        x31 = self.cross_attention_block31(x1, x2)
        x32 = self.cross_attention_block32(x1, x2)

        z = self.fused_attention_block(x12,x13,x21,x23,x31,x32,y)

        query, positive, negative = z[:,0,:], z[:,1,:], z[:,1:,:]

        query = query.squeeze(1)
        positive = positive.squeeze(1)

        return z, query, positive, negative




class ContrastDataset(data.Dataset):
    def __init__(self,  path, device):
        df = pd.read_csv(path,sep=',', header=0, dtype={'anchor':int,'positive':int, 'negative':str})
        df['negative'] = df['negative'].apply(lambda x : eval(x))
        self.device = device
        self.data = df
        self.data = self.data.values

        
    def __getitem__(self, index):
        anchor, pos, negative = self.data[index]
        data = [anchor, pos] + negative
        data = torch.IntTensor(data).to(self.device)
        
        return data
    
    def __len__(self):
        return len(self.data)
        
    


if __name__ == "__main__":

    path1 = "./Embed/LLM_Embed/NY/NY_llama2_time_LAST.pt"
    path2 = "./Embed/LLM_Embed/NY/NY_llama2_address_LAST.pt"
    path3 = "./Embed/LLM_Embed/NY/NY_llama2_cat_nearby_LAST.pt"
    path4 = "./Embed/Poi_Model_Embed/tale_256_ny/poi_repr/poi_repr.pth"

    device = 'cuda:2'
    batch_size = 64
    EPOCH = 200

    train_dataset = ContrastDataset('./ContrastDataset/train.csv', device)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    # LLM_cat_neaby_embed = torch.load("./Embed/LLM_Embed/NY/NY_llama2_cat_nearby_LAST.pt").to('cuda:0')
    # LLM_address_embed = torch.load("./Embed/LLM_Embed/NY/NY_llama2_address_LAST.pt")
    # LLM_time_embed = torch.load("./Embed/LLM_Embed/NY/NY_llama2_time_LAST.pt")

    # POI_embed = torch.load("./Embed/Poi_Model_Embed/tale_128_ny/poi_repr/poi_repr.pth")



    Model = PoiEnhancer(path1, path2, path3, path4).half().cuda(device)
    Model.train()
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)

    
    nceloss = InfoNCE(negative_mode='paired')
   

    for e in tqdm(range(EPOCH)):
        for batch in train_dataloader:
            _, query, positive, negative = Model(batch)
            loss = nceloss(query, positive, negative)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()






    


    

   

    
        
    

    # LLM_embedding_layer = 