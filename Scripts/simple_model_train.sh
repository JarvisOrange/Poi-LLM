#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset  NY \
    --poi_model tale \
    --LLM llama3 \
    --gpu 2 \
    --epoch_num 50 \
    --batch_size 128 \
    --dim 256 \
    --save_interval 3