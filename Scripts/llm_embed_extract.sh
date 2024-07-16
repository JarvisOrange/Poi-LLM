#!/bin/bash

for variable1 in 1
    do
    for variable2 in 'address' 'time' 'cat_nearby'
    do  
        python Tools/get_embedding_from_LLM.py  --LLM chatglm3 --dataset NY --gpu 3 --prompt_type $variable2
    done
done