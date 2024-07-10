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
        if self.dim_reduct == True: 
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


# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(dim))
#         self.register_buffer("beta", torch.zeros(dim))

#     def forward(self, x):
#         return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


    
class SingleAttentionBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 dim_fused,
                 dim_head=32, 
                 heads=1,
                 ff_mult=4
                 ):

        super().__init__()
        self.scale = dim_head ** -0.5

        inner_dim =  dim_head * heads
        self.heads = heads

        self.norm = nn.LayerNorm(dim, eps=1e-4)
        self.norm_ = nn.LayerNorm(dim_fused,eps=1e-4)
        

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_head, bias=False)


        ff_inner_dim = ff_mult * dim_head

        self.feedforward = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim_head, bias=False)
        )

    

    def forward(self, x, y):
        """ 
        b - batch
        d - feature dimension
        """

        # x_ = x

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

        self.norm = nn.LayerNorm(dim, eps=1e-4)
        self.norm_ = nn.LayerNorm(dim_fused, eps=1e-4) if dim_fused else nn.Identity()

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

        return out + x_
    

    
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

        self.norm = nn.LayerNorm(dim, eps=1e-4)

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



        return out + x_
        

 
        


    

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

        self.fuse_attention =  FuseAttentionBlock(dim=self.poi_e_dim, dim_fused= 2 * self.poi_e_dim)

        self.cross_attention_block = CrossAttentionBlock(dim=self.poi_e_dim, dim_fused=self.poi_e_dim)



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

        z = self.cross_attention_block(temp_out, y)
        
        return z




class ContrastDataset(data.Dataset):
    def __init__(self,  path, device):
        df = pd.read_csv(path,sep=',', header=0, dtype={'anchor':int,'positive':int, 'negative':str})
        df = df.sample(frac=0.005)
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

class PoiDataset(data.Dataset):
    def __init__(self,  path, device):
        df = pd.read_csv(path,sep=',', header=0, usecols=['geo_id','type'])
        df = df[df['type']=='Point']
        df = df.drop(['type'], axis=1)
        first = df.iloc[0,0]
        df['geo_id'] = df['geo_id'].apply(lambda x: x - first)

        self.device = device
        self.data = df
        self.data = self.data.values

        
    def __getitem__(self, index):
        data = self.data[index]
        data = torch.IntTensor(data).to(self.device)
        return data
    
    def __len__(self):
        return len(self.data)
    

        
    


if __name__ == "__main__":

    path1 = "./Embed/LLM_Embed/NY/NY_llama2_time_LAST.pt"
    path2 = "./Embed/LLM_Embed/NY/NY_llama2_address_LAST.pt"
    path3 = "./Embed/LLM_Embed/NY/NY_llama2_cat_nearby_LAST.pt"
    path4 = "./Embed/Poi_Model_Embed/tale_256_ny/poi_repr/poi_repr.pth"

    device = 'cuda:0'
    batch_size = 512
    EPOCH = 200

    train_dataset = ContrastDataset('./ContrastDataset/train.csv', device)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)




    Model = PoiEnhancer(path1, path2, path3, path4).half().cuda(device)
    Model.train()
    # optimizer = torch.optim.Adam(Model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = torch.optim.Adam(Model.parameters(), lr=1e-3)

    
    nceloss = InfoNCE(negative_mode='paired')



    for e in tqdm(range(EPOCH)):
        l = []
        for _, batch in enumerate(train_dataloader):
        
            z  = Model(batch)
            query, positive, negative = z[:,0,:], z[:,1,:], z[:,1:,:]
            query = query.squeeze(1)
            positive = positive.squeeze(1)
            loss = nceloss(query, positive, negative)

            optimizer.zero_grad()

            print(loss.item())

            l.append(loss.item())

            

            for name, parms in Model.cross_attention_block.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")

            loss.backward()

            print("________________----------------________")
            
            for name, parms in Model.cross_attention_block.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
            
            


            optimizer.step()
            print("________________----------------________")
            for name, parms in Model.cross_attention_block.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
            if _ ==0:
                exit()
            

        print(sum(l)/ len(l))


        

    model_path =  "./Model_cache/model" + str(EPOCH) +".pth"

    torch.save({'model': Model.state_dict()}, model_path)

    #########save embed ################
    embed_path = './Embed/Result_Embed/'+"temp.pt"

    poi_dataset = PoiDataset('./Dataset/Foursquare_NY/ny.geo', device)
    poi_dataloader = DataLoader(poi_dataset, batch_size = batch_size, shuffle=False)

    result_embed = torch.empty((len(poi_dataset), 256)).cpu()
    torch.cuda.empty_cache()
    index = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(poi_dataloader)):
            
            # batch = batch.unsqueeze(dim=1)
            out = Model(batch)
            out = out.squeeze(1)


            if out.shape[0] !=  batch_size:
                result_embed[step * batch_size :step * batch_size + out.shape[0],:] = out.cpu()
                index +=  out.shape[0]
            else:
                result_embed[step * batch_size : (step +1)* batch_size,:] = out.cpu()
                index +=  batch_size

    torch.save(result_embed, embed_path)


