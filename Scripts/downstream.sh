#!/bin/bash

n="NY_llama2_multilayer_tale_256_Epoch_40"

python Downstream/poi_clf.py  --gpu 2 --NAME $n

echo "----------- task 1 done -----------"

python Downstream/traj_next_pre.py   --gpu 2 --NAME $n

echo "----------- task 2 done -----------"
