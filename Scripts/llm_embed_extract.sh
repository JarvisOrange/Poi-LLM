#!/bin/bash

for variable1 in 1
    do
    for variable2 in   'cat_nearby'
    do  
        python Tools/get_embedding_from_LLM.py  --LLM llama2 --dataset SG --gpu 3 --prompt_type $variable2
    done
done

