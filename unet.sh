#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet.py \
#--epoch=10 \
#--modality=flow \
#--lr=5e-4 \
#--restore=result/0901_1437_unet_flow/0901_1437_unet_flow_epoch15_model.pth.tar \
#--dataset=ucf \
#--clip-num=6 \
#--lr-policy=ucf2 \
#--batch-size=12 \
#--num-worker=16
#wait
#sleep 20
CUDA_VISIBLE_DEVICES=0,1,2,3 python unet.py \
--epoch=400 \
--modality=flow \
--restore=result/0903_2148_unet_rgb/0903_2148_unet_rgb_epoch79_model.pth.tar \
--dataset=thumos \
--optimizer=sgd \
--lr-policy=thumos \
--lr=1e-3 \
--clip-num=4 \
--eval-freq=400 \
--print-freq=4 \
--batch-size=96 \
--num-worker=16
#wait
#sleep 20
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet.py \
#--epoch=20 \
#--modality=flow \
#--restore=result/0903_0659_unet_flow/0903_0659_unet_flow_epoch19_model.pth.tar \
#--dataset=thumos \
#--optimizer=sgd \
#--evaluate \
#--lr=1e-4 \
#--clip-num=4 \
#--eval-freq=25 \
#--print-freq=1 \
#--batch-size=2 \
#--num-worker=2


#--exp-name=0901_rgb_second
#wait
#sleep 20
#CUDA_VISIBLE_DEVICES=0,1,2,3 python unet.py \
#--epoch=1 \
#--modality=rgb \
#--restore=result/0902_1226_unet_rgb/0902_1226_unet_rgb_epoch0_model.pth.tar \
#--dataset=ucf \
#--lr-policy=ucf3 \
#--clip-num=6 \
#--lr=1e-5 \
#--batch-size=12 \
#--num-worker=24