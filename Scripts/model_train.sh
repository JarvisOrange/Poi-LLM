#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset  SG \
    --poi_model teaser \
    --LLM llama2 \
    --gpu 1 \
    --epoch_num 100 \
    --batch_size 128 \
    --dim 256 \
    --save_interval 5