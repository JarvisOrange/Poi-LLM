#!/bin/bash

n="NY_llama2_multilayer_tale_256_Epoch_40"
pn='tale_256_ny'

# python Downstream/poi_clf.py  --gpu 1 --NAME $n --dataset NY --POI_MODEL_NAME $pn

# echo "----------- task 1 done -----------"

python Downstream/traj_next_pre.py   --gpu 1 --NAME $n --dataset NY --POI_MODEL_NAME $pn

echo "----------- task 2 done -----------"
