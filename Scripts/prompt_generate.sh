#!/bin/bash

for variable1 in 'NY' 
    do
    for variable2 in  'cat_nearby' 'address' 'time'
    do  
        python Tools/gen_ablation_prompt.py  --dataset $variable1 --prompt_type $variable2
    done
done