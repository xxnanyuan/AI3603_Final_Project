#!/bin/bash
GAMMA_LIST=("0.2")
LR_LIST=("0.0005")
for GAMMA in ${GAMMA_LIST[@]}
do
    for LR in ${LR_LIST[@]}
    do
        python trainRacetrack_scsac.py --env racetrack-v0 --gamma $GAMMA --policy-lr $LR --q-lr $LR --buffer-size 1000000 --total-timesteps 250000 --target-network-frequency 1 --batchsize 256
    done 
done



