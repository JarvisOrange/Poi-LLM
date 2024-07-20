import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset

from tqdm import tqdm
from loguru import logger

import einops
from transformers import AutoTokenizer, AutoModel


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="gpu")

    parser.add_argument(
        "--LLM",
        type=str,
        default="llama2",
        choices=["llama2", "llama3", "chatglm2","chatglm3", "gpt2","gpt2_medium","gpt2_large","gpt2_xl"],
        help="which LLM to use",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="NY",
        choices=["NY", "SG", "TKY"],
        help="which dataset",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="time",
        choices=['address', 'time','cat_nearby'],
    )

    args = parser.parse_args()

    return args


def main():
    args = create_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


    LLM_datapath = "./LLMs/"+args.LLM

    tokenizer = AutoTokenizer.from_pretrained(LLM_datapath, trust_remote_code=True)
 
    model = AutoModel.from_pretrained(LLM_datapath, trust_remote_code=True).half().cuda(device)
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<sop>'})
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    model = model.eval()

    prompt_data_path = "./Prompt/" + "" + args.dataset +"/"+ "prompt_" + args.dataset + "_" + args.prompt_type + '.csv'


    prompt_df = pd.read_csv(prompt_data_path, header=0)
    print(len(prompt_df))
    dataset_strings = list(prompt_df['prompt'])

    

    tk_result = tokenizer.batch_encode_plus(
            dataset_strings,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    token_ids = tk_result['input_ids']

    token_ids = torch.cat([torch.ones(token_ids.shape[0], 1, dtype=torch.long) * tokenizer.bos_token_id, token_ids], dim=1)


    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
    entity_mask[:, prompt_tokens] = False
    entity_mask[token_ids == tokenizer.pad_token_id] = False


    tk_dataset = Dataset.from_dict({
            'input_ids': token_ids.tolist(),
            'entity_mask': entity_mask.tolist(),
        })

    tk_dataset.set_format(type='torch', columns=['input_ids'])

    #  activation extraction

    def process_activation_batch(batch_activations, step, batch_mask=None):
        if args.LLM == 'chatglm3' or args.LLM == 'chatglm2':
            batch_activations = einops.rearrange(batch_activations, 'n b d -> b n d')

        cur_batch_size = batch_activations.shape[0]

        last_ix = batch_activations.shape[1] - 1
        batch_mask = batch_mask.to(int)
        last_entity_token = last_ix - \
            torch.argmax(batch_mask.flip(dims=[1]), dim=1)

        

        d_act = batch_activations.shape[2]
        expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
        
        processed_activations = batch_activations[
            torch.arange(cur_batch_size).unsqueeze(-1),
            expanded_mask,
            torch.arange(d_act)
        ]

        assert processed_activations.shape == (cur_batch_size, d_act)

        return processed_activations
    

    with torch.no_grad():
        if args.LLM == 'llama2' or args.LLM =='llama3':
            layers = list(range(model.config.num_hidden_layers))
        elif  args.LLM == 'chatglm2' or args.LLM == 'chatglm3':
            layers = list(range(model.config.num_layers))
        elif  args.LLM == 'gpt2' or args.LLM == 'gpt2_large'or args.LLM == 'gpt2_medium' or args.LLM == 'gpt2_xl':
            layers = list(range(model.config.n_layer))

        entity_mask = torch.tensor(tk_dataset['entity_mask'])

        n_seq, ctx_len = token_ids.shape

        activation_rows = n_seq

        layer_activations = {
                l: torch.zeros(activation_rows, model.config.hidden_size,
                            dtype=torch.float16)
                for l in layers
            }
        
        offset = 0
        bs = 32
        layer_offsets = {l: 0 for l in layers}
        dataloader = DataLoader(tk_dataset['input_ids'], batch_size=bs, shuffle=False)
        
        for step, batch in enumerate(tqdm(dataloader, disable=False)):
            batch_entity_mask = entity_mask[step*bs:(step+1)*bs]
            
            last_valid_ix = torch.argmax(
                    (batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
            
            batch = batch[:, :last_valid_ix].to(device)
            batch_entity_mask = batch_entity_mask[:, :last_valid_ix]
  
            out = model(batch, output_hidden_states=True,
                        output_attentions=False, return_dict=True, use_cache=False)

            
            for lix, activation in enumerate(out.hidden_states[1:]):
                if lix not in layer_activations:
                    continue
                activation = activation.cpu().to(torch.float16)
            
                processed_activations = process_activation_batch(
                        activation, step, batch_entity_mask)
                
                save_rows = processed_activations.shape[0]
                
                layer_activations[lix][offset:offset +save_rows] = processed_activations


            offset += batch.shape[0]


    last_layer_id = layers[-1]

    last_layer_activation = layer_activations[last_layer_id]
        
    save_path = "./Embed/LLM_Embed/" + "" + args.dataset +"/"    
    save_name = f'{args.dataset}_{args.LLM}_{args.prompt_type}_LAST.pt'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, save_name)

    torch.save(last_layer_activation, save_path)


if __name__ == "__main__":
    main()