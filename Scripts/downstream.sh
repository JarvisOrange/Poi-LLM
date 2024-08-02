#!/bin/bash

n="NY_llama2_ctle_256_Epoch_45"
pn='ctle_256_ny'
dataset='NY'
gpu=3

echo $n

# python Downstream/poi_clf.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 1 done -----------"


# python Downstream/traj_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 2 done -----------"

# python Downstream/flow_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 3 done -----------"


python Downstream/traj_user_clf.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

echo $n

echo "----------- task 4 done -----------"

python Downstream/poi_cluster.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

echo $n

echo "----------- task 5 done -----------"

