#!/bin/bash


for variable1 in 1
    do
    for variable2 in    'sum'
    do  
        python Tools/get_embedding_from_LLM.py  --LLM llama2 --dataset NY --gpu 1 --prompt_type $variable2
    done
done


