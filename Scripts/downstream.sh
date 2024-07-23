#!/bin/bash

n="SG_llama2_tale_256_Epoch_5"
pn='tale_256_sg'
dataset='SG'

echo $n

# python Downstream/poi_clf.py    --gpu 3 --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 1 done -----------"



python Downstream/traj_next_pre.py    --gpu 3 --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

echo $n

echo "----------- task 2 done -----------"

