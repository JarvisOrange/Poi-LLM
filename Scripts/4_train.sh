#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset SG \
    --poi_model skipgram \
    --LLM llama2 \
    --gpu 3 \
    --epoch_num 100 \
    --dim 256 \
    --save_interval 5