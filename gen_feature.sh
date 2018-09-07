#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,3 nohup python gen_feature.py \
> 0831_gen.log 2>&1 &