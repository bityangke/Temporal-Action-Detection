#!/usr/bin/env bash
nohup python train_act_i3d.py \
--epoch=12  --num-workers=32 --batch-size=32 > 0813_102act.log 2>&1 &
