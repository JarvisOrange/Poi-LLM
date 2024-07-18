#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset NY \
    --poi_model ctle \
    --LLM llama3 \
    --gpu 0 \
    --epoch_num 50 \
    --dim 256 \
    --save_interval 5