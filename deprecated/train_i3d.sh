#!/usr/bin/env bash
nohup python train_i3d.py \
--epoch=12  --restore=result/0825_0001_unet_model.pth.tar --start-epoch=10\
> 0825_unet_ucf.log 2>&1 &