#!/bin/bash

n="NY_llama2_dualoss_tale_256_Epoch_10"
pn='tale_256_ny'



python Downstream/poi_clf.py    --gpu 3 --NAME $n --dataset NY --POI_MODEL_NAME $pn

echo "----------- task 1 done -----------"


python Downstream/traj_next_pre.py    --gpu 3 --NAME $n --dataset NY --POI_MODEL_NAME $pn > downstream_log.txt


echo "----------- task 2 done -----------"
