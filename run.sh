#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
nohup bash unet.sh > 0904_th_unet_flow.log 2>&1 &



#export CUDA_VISIBLE_DEVICES=3
#nohup python train_classifier.py \
#--epoch=3000 \
#--unet-softmax='in' \
#--modality=flow \
#--dataset=thumos \
#--clip-gradient=5 \
#--lr=1e-3 \
#--restore=result/0831_2100_unet_flow/0831_2100_unet_flow_epoch499_model.pth.tar \
#--eval-freq=500 \
#--num-worker=4 \
#--clip-num=8 \
# > 0831_unet_flow3.log 2>&1 &


#result/0831_2101_unet_rgb/0831_2101_unet_rgb_epoch699_model.pth.tar
#
#FLOW:
#
#result/0831_2100_unet_flow/0831_2100_unet_flow_epoch499_model.pth.tar

#--evaluate \
#--restore=result/0830_1952_unet_rgb/0830_1952_unet_rgb_epoch999_model.pth.tar \


#--evaluate \
#--restore=result/0830_1952_unet_rgb/0830_1952_unet_rgb_epoch1999_model.pth.tar \ss