#!/bin/bash
GAMMA_LIST=("0.8")
LR_LIST=("0.001")
for GAMMA in ${GAMMA_LIST[@]}
do
    for LR in ${LR_LIST[@]}
    do
        python trainRacetrack.py --env racetrack-v0 --gamma $GAMMA --policy-lr $LR --q-lr $LR --buffer-size 10000
    done 
done