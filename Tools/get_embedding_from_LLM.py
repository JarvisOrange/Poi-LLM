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

from tqdm import tqdm
from loguru import logger

import einops
from transformers import AutoTokenizer, AutoModel

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument(
        "--LLM",
        type=str,
        default="llama2",
        choices=["llama2", "llama3", "chatglm2", "chatglm2-6b", "gpt2","gpt2-medium","gpt2-large","gpt2-xl"],
        help="which LLM to use",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="instagram_NY",
        choices=["instagram_NY", "foursquare_NY", "foursquare_TKY"],
        help="which dataset",
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="None",
        choices=["lonlat", 'address', 'time','category'],
        help="which LLM to use",
    )

    args = parser.parse_args()

    return args


def main():
    args = create_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.LLM, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.LLM, trust_remote_code=True).half().cuda(device)
    model = model.eval()

    dataset_dir = "./Dataset/" + args.dataset + "POI.csv"

    PROMPTS = {
        'lonlat': ' ',
        'address': '',
        'time': ' ',
        'poi': '  ',
    }

    poi_info =  pd.read_csv("../dataset/instagram_data/poi_index.txt", sep='\t')

    oldpoi_df = pd.read_csv("../dataset/instagram_data/train_content.txt", sep='\t', header=None)

    poi_df = poi_info.iloc[oldpoi_df.iloc[:,0],:]
    poi_df.columns = ['Index','NAME']
    dataset_strings = [ PROMPTS['where_is'] + poi_info.iloc[poi_index,1] for poi_index in list(oldpoi_df.iloc[:,0]) ]

    tk_result = tokenizer.batch_encode_plus(
            dataset_strings,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    token_ids = tk_result['input_ids']

    token_ids = torch.cat([
        torch.ones(token_ids.shape[0], 1,
                    dtype=torch.long) * tokenizer.bos_token_id,
        token_ids], dim=1
    )


    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
    entity_mask[:, prompt_tokens] = False
    entity_mask[token_ids == tokenizer.pad_token_id] = False


    tk_dataset = datasets.Dataset.from_dict({
            'entity': poi_df['NAME'].to_list(),
            'input_ids': token_ids.tolist(),
            'entity_mask': entity_mask.tolist(),
        })

    tk_dataset.set_format(type='torch', columns=['input_ids'])

    #  activation extraction

    def process_activation_batch(batch_activations, step, batch_mask=None):
        batch_activations = einops.rearrange(batch_activations, 'c b d -> b c d')
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
        
        layers = list(range(model.config.num_layers))
        entity_mask = torch.tensor(tk_dataset['entity_mask'])

        n_seq, ctx_len = token_ids.shape


        activation_rows = n_seq

        layer_activations = {
                l: torch.zeros(activation_rows, model.config.hidden_size,
                            dtype=torch.float16)
                for l in layers
            }
        offset = 0
        bs = 128
        layer_offsets = {l: 0 for l in layers}
        dataloader = DataLoader(tk_dataset['input_ids'], batch_size=bs, shuffle=False)
        
        for step, batch in enumerate(tqdm(dataloader, disable=False)):
            # clip batch to remove excess padding
            batch_entity_mask = entity_mask[step*bs:(step+1)*bs]
            
            last_valid_ix = torch.argmax(
                    (batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
            
            batch = batch[:, :last_valid_ix].to(device)
            batch_entity_mask = batch_entity_mask[:, :last_valid_ix]
            batch[batch_entity_mask] = model.config.mask_token_id
            out = model(batch, output_hidden_states=True,
                        output_attentions=False, return_dict=True, use_cache=False)

            # do not save post embedding layer activations
            for lix, activation in enumerate(out.hidden_states[1:]):
                if lix not in layer_activations:
                    continue
                activation = activation.cpu().to(torch.float16)
                processed_activations = process_activation_batch(
                        activation, step, batch_entity_mask)
                

                save_rows = processed_activations.shape[0]
                if offset + save_rows > activation_rows:
                    print(batch.shape)
                    print(offset + save_rows)
                    print(processed_activations.shape)
                    layer_activations[lix][offset:offset +
                                        save_rows] = processed_activations
                else:
                    layer_activations[lix][offset:offset +
                                        save_rows] = processed_activations


            offset += batch.shape[0]

    for layer_ix, activations in layer_activations.items():
        
        activation_save_path = "./result/instagram_NY_glm_6b/"
        if not os.path.exists(activation_save_path):
            os.makedirs(activation_save_path)
        # save_name = f'{args.entity_type}.{args.activation_aggregation}.{prompt_name}.{layer_ix}.pt'
        save_name = f'NY_POI_LAST.{layer_ix}.pt'
        save_path = os.path.join(activation_save_path, save_name)
        # activations = adjust_precision(
        #     activations.to(torch.float32), args.save_precision, per_channel=True)
        torch.save(activations, save_path)


    


if __name__ == "__main__":
    main()