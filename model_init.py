import torch
from torch import einsum, nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pandas as pd

from einops import rearrange, repeat


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


class POI_LLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        unimodal_depth,
        multimodal_depth,
        dim_latents = None,
        image_dim = None,
        num_img_queries=256,
        dim_head=64,
        heads=8,
        ff_mult=4,
        contrastive_loss_weight=1.,
        pad_id=0
    ):
        super().__init__()
        self.dim = dim

        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # token embeddings

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.text_cls_token = nn.Parameter(torch.randn(dim))

        # image encoder

        self.img_encoder = img_encoder

        # attention pooling for image tokens

        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)

        self.img_attn_pool_norm = LayerNorm(dim)
        self.text_cls_norm = LayerNorm(dim)

        # to latents

        dim_latents = default(dim_latents, dim)
        self.img_to_latents = EmbedToLatents(dim, dim_latents)
        self.text_to_latents = EmbedToLatents(dim, dim_latents)

        # contrastive learning temperature

        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # unimodal layers

        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
            )

        # multimodal layers

        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
            ]))

        # to logits

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

        # whether in data parallel setting
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def embed_text(self, text):
        batch, device = text.shape[0], text.device

        seq = text.shape[1]

        text_tokens = self.token_emb(text)

        # append text cls tokens

        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # create specific mask for text cls token at the end
        # to prevent it from attending to padding

        cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through unimodal layers

        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # get text cls token

        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        text_embeds = self.text_cls_norm(text_cls_tokens)
        return text_embeds, text_tokens

    def embed_image(self, images=None, image_tokens=None):
        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image tokens

        assert not (exists(images) and exists(image_tokens))

        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            image_tokens = self.img_encoder(images)

        # attention pool image tokens

        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]

    def forward(
        self,
        batch,
        
        labels=None,
        return_loss=False,
        return_embeddings=False
    ):
        batch, device = text.shape[0], text.device

        if return_loss and not exists(labels):
            text, labels = text[:, :-1], text[:, 1:]

        text_embeds, text_tokens = self.embed_text(text)

        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        # return embeddings if that is what the researcher wants

        if return_embeddings:
            return text_embeds, image_embeds

        # go through multimodal layers

        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)

        logits = self.to_logits(text_tokens)

        if not return_loss:
            return logits

        # shorthand

        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)

        logits = rearrange(logits, 'b n c -> b c n')
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # embedding to latents

        text_latents = self.text_to_latents(text_embeds)
        image_latents = self.img_to_latents(image_embeds)

        # maybe distributed all gather

        if self.is_distributed:
            latents = torch.stack((text_latents, image_latents), dim = 1)
            latents = all_gather(latents)
            text_latents, image_latents = latents.unbind(dim = 1)

        # calculate contrastive loss

        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss



class FusedAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, head, ff_mult):
        self.heads = heads
        self.scale = 
        
        self.to_q = nn.Linear()

        self.


        self.feedforward = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

    def forward(self, x1, x2, x3, x4, x5, x6, y):
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

        self.norm = LayerNorm(dim)
        self.norm_ = LayerNorm(dim_fused) if dim_fused else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_fused, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
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

        self.ff = nn.Sequential(
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
    def __init__(self, hidden_dim=256, embed_path='', dim_reduct=True):
        self.embed = torch.load(embed_path)
        num, dim = self.embed.shape

        self.embedding_layer = nn.Embedding(num, dim, _weight=self.embed)
        
        self.dim_reduct = dim_reduct
        if self.dim_reduct: 
            self.embed2hidden = Embed2hidden(self.dim_embed, dim)
        else:
            self.embed2hidden = nn.Identity()
        

    def forward(self, x):
        with torch.no_grad():
            x = self.embedding_layer(x)
        out = self.embed2hidden(x)
        return out
 
class PoiEnhancer(nn.Module):
    def __init__(self, 
                 poi_repr_embedding_layer,
                 llm_embedding_layer_tuple
                 ):
        
        super().__init__()
        self.llm_layer1, self.llm_layer2, self.llm_layer3 = llm_embedding_layer_tuple
        self.poi_layer = poi_repr_embedding_layer

        self.EmbeddingBlock

        self.attention_block1 = AttentionBlock(dim_in, dim_out, head, ff_mult)
        self.attention_block2 = AttentionBlock()
        self.attention_block3 = AttentionBlock()

        self.cross_attention_block12 = CrossAttentionBlock() 
        self.cross_attention_block13 = CrossAttentionBlock() 
        self.cross_attention_block21 = CrossAttentionBlock() 
        self.cross_attention_block23 = CrossAttentionBlock() 
        self.cross_attention_block31 = CrossAttentionBlock() 
        self.cross_attention_block32 = CrossAttentionBlock() 

        self.fused_attention_block = FusedAttentionBlock()


        

    def forward(self, batch):
        with torch.no_grad():
            llm_e1 = self.llm_layer1(batch)
            llm_e2 = self.llm_layer1(batch)
            llm_e3 = self.llm_layer1(batch)

            p_e = self.poi_layer(batch)

        
        
    


if __name__ == "__main__":
    LLM_cat_neaby_embed = torch.load("./Embed/LLM_Embed/NY/NY_llama2_cat_nearby_LAST.pt").to('cuda:0')
    LLM_address_embed = torch.load("./Embed/LLM_Embed/NY/NY_llama2_address_LAST.pt")
    LLM_time_embed = torch.load("./Embed/LLM_Embed/NY/NY_llama2_time_LAST.pt")

    POI_embed = torch.load("./Embed/Poi_Model_Embed/tale_128_ny/poi_repr/poi_repr.pth")

    poi_num, dim = POI_embed.shape

    # x = LLM_address_embed.shape
    z = LLM_cat_neaby_embed.shape

    poi_repr_embedding_layer = nn.Embedding(poi_num, dim, _weight=POI_embed)

    llm_embedding_layer = nn.Embedding(poi_num, dim, _weight=LLM_cat_neaby_embed)

    
        
    

    # LLM_embedding_layer = 