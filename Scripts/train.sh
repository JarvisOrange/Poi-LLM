#!/bin/bash
python main.py --dataset SG --poi_model ctle --LLM llama3 --gpu 3 --epoch_num 100 --dim 256 --save_interval 5 --simple True
