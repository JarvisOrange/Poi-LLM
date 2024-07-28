#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset  TKY \
    --poi_model ctle \
    --LLM llama2 \
    --gpu 3 \
    --epoch_num 100 \
    --batch_size 128 \
    --dim 256 \
    --save_interval 5 \