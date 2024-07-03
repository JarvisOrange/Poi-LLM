#!/bin/bash

for variable1 in 'NY' 'SG' 'TKY'
    do
    for variable2 in 'address' 'time' 'cat_nearby'
    do  
        python Tools/gen_prompt.py  --dataset $variable1 --prompt_type $variable2
    done
done