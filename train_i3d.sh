#!/usr/bin/env bash
nohup python train_i3d.py \
--epoch=10  \
> 0804_i3d.log 2>&1 &