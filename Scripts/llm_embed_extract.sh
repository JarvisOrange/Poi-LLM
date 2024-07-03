#!/bin/bash

for variable1 in 1 2 3
    do
    for variable2 in 'address' 'time' 'cat_nearby'
    do  
        python Tools/get_embedding_from_LLM  --LLM llama2 --dataset NY --gpu --prompt_type $variable2
    done
done