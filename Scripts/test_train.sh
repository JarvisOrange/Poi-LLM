#!/bin/bash


#!/bin/bash
python main.py \
    --simple_dataset True \
    --dataset TKY \
    --poi_model ctle \
    --LLM llama2 \
    --gpu 2 \
    --epoch_num 50 \
    --dim 256 \
    --save_interval 1 \
    
    
# CUDA_VISIBLE_DEVICES="0, 1, 2, 3" \
# python -m torch.distributed.run 
#     --nproc_per_node 4
#     --simple_dataset False \
#     --dataset SG \
#     --poi_model teaser \
#     --LLM llama3 \
#     --gpu 0 \
#     --epoch_num 50 \
#     --dim 256 \
#     --save_interval 1
#     --batch_size 64
#     --DDP True \
#     --local_rank 0 \
#     main.py 