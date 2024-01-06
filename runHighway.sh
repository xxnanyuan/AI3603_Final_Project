#!/bin/bash
GAMMA_LIST=("0.95")
LR_LIST=("0.001")
for GAMMA in ${GAMMA_LIST[@]}
do
    for LR in ${LR_LIST[@]}
    do
        python trainHighway.py --env highway-v0 --gamma $GAMMA --policy-lr $LR --q-lr $LR --buffer-size 50000 --total-timesteps 200000 --target-network-frequency 2 --batchsize 256
    done 
done