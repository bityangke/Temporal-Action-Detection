#!/usr/bin/env bash\
CUDA_VISIBLE_DEVICES=3 nohup python main.py \
--use-gae --num-steps=5 --log-interval=50 --num-processes=16 --num-frames=1000000 --num-mini-batch=64 --exp-name=0807_rl_3 \
> 0807_rl_3.log 2>&1 &