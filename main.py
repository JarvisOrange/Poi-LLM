import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import einsum, nn
from torch.autograd import Function
from einops import rearrange, repeat
from torch.utils import data
from torch.utils.data import DataLoader	
from info_nce import InfoNCE, info_nce


from poi_utils import *
from model_init import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="NY",
        choices=["NY","SG","TKY"],
        help="which dataset",
    )
    parser.add_argument(
        "--poi_model", type=str, default='tale', 
    )  # very sensitive 
    parser.add_argument(
        "--LLM", type=str, default='llama2')
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu"
    )  # batch size
    parser.add_argument(
        "--epoch_num", type=int, default=100, help="epoch number")
    
    parser.add_argument(
        "--dim", type=int, default=256, help="model dimensions")
    
    parser.add_argument(
        "--lr", type=float, default=1e-3)
    
    parser.add_argument(
        "--epoch", type=int, default=100)
    
    parser.add_argument(
        "--batch_size", type=int, default=128)
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
    )
    
    parser.add_argument(
        "--cross_layer_num",
        type=int,
        default=3,
        help="depth of the unimodal transformer",
    )

    parser.add_argument(
        "--simple_dataset",
        type=str,
        default='False'
    )

    parser.add_argument(
        '--DDP',
        default = 'False'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default = -1
    )

    parser.add_argument(
        '--cpu',
        type=int,
        default = 0
    )

    parser.add_argument(
        '--ablation',
        type=int,
        default = 0
    )

    args = parser.parse_args()

    return args


def main():
    args = create_args()
    poi_model = args.poi_model
    LLM = args.LLM
    dataset = args.dataset
    dim = args.dim
    device = 'cuda:' +str(args.gpu)
    
    ablation = args.ablation

    LR = args.lr
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    SAVE_INTERVAL = args.save_interval

    cross_layer_num = args.cross_layer_num

    if ablation == 1:
        llm_embed_path = "./Washed_Embed/Ablation_Embed/" + dataset
    else:
        llm_embed_path = "./Washed_Embed/LLM_Embed/" + dataset

    poi_embed_path = "./Washed/"

    llm_name_list_address = [dataset, LLM, 'address','LAST']
    llm_name_list_cat_nearby = [dataset, LLM, 'cat_nearby','LAST']
    llm_name_list_time = [dataset, LLM, 'time','LAST']

    poi_name_list = [poi_model, str(dim), dataset.lower()]
    

    path_address = llm_embed_path + '/' + '_'.join(llm_name_list_address) + '.pt'
    path_cat = llm_embed_path + '/' + '_'.join(llm_name_list_cat_nearby) + '.pt'
    path_visit = llm_embed_path + '/' + '_'.join(llm_name_list_time) + '.pt'


    path4 =  poi_embed_path +'/' + '_'.join(poi_name_list) + '/poi_repr.pth' 


    train_data_name = dataset+'_train.csv'
    
    train_dataset = ContrastDataset('./Washed_ContrastDataset/' + train_data_name, device, simple=args.simple_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)


    Model = PoiEnhancer(path_address, path_cat, path_visit, path4, cross_layer_num=cross_layer_num, dim=dim).to(device)
    Model.train()
    optimizer = torch.optim.AdamW(Model.parameters(), lr=LR, weight_decay=1e-3)

    
    nceloss = InfoNCE(temperature=0.1,reduction='mean',negative_mode='paired')


    Model.train()
    
    for epoch in range(EPOCH):
        l = []
        for batch in tqdm(train_dataloader):
            
            z, y  = Model(batch)

            query, positive, negative = z[:,0,:], z[:,1,:], z[:,1:,:]

            
            query_ = query.squeeze(1)
            positive_ = positive.squeeze(1)

            z = rearrange(z, 'b n d -> (b n) d')
            y = rearrange(y, 'b n d -> (b n) d')

    
            loss = nceloss(query_, positive_, negative) + simloss(z, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            l.append(loss.item())

            
        print('epoch %d, loss： %.4f' % (epoch+1,sum(l)/ len(l)))
    

        if (epoch + 1) % SAVE_INTERVAL == 0:
            Model.eval()
            save_embed(Model, dataset, LLM, dim, poi_model, epoch+1, device, train_split=False, ablation=ablation)
            Model.train()

        optimizer.zero_grad()

        #########save embed ################







if __name__ == "__main__":
    main()

