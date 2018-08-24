#!/usr/bin/env bash\
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python gen_feature.py \
> 0813_gen_act3.log 2>&1 &
