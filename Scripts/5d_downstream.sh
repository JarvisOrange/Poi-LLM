#!/bin/bash

n="SG_llama2_teaser_256_Epoch_50"
pn='teaser_256_sg'
dataset='SG'
gpu=0

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

python Downstream/traj_next_pre.py    --gpu $gpu --NAME $n --dataset $dataset --POI_MODEL_NAME $pn --epoch 100

echo $n

echo "----------- task 5 done -----------"

