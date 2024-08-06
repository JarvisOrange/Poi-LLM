#!/bin/bash

n="TKY_llama2_skipgram_256_Epoch_50"
pn='skipgram_256_tky'
dataset='TKY'
gpu=1

echo $n

# python Downstream/poi_clf.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 1 done -----------"

# python Downstream/poi_cluster.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn


# echo $n

# echo "----------- task 2 done -----------"

# python Downstream/flow_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 3 done -----------"


# python Downstream/traj_user_clf.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

# echo $n

# echo "----------- task 4 done -----------"

python Downstream/traj_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn

echo $n

echo "----------- task 5 done -----------"

