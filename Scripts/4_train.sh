#!/bin/bash
python main.py \
    --simple_dataset False \
    --dataset NY \
    --poi_model poi2vec \
    --LLM llama2 \
    --gpu 1 \
    --epoch_num 50 \
    --dim 256 \
    --save_interval 5