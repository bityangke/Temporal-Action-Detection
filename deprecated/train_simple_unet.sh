#!/usr/bin/env bash
nohup python train_simple_unet.py \
--epoch=3 \
--start-epoch=2 \
--batch-size=32 \
--restore=result/0826_0223_unet_model.pth.tar \
--num-worker=12 \
--lr=1e-5 \
> 0826_unet_t5.log 2>&1 &