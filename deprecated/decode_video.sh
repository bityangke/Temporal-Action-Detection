#!/bin/bash

set -e
set -x

mkdir -p "/nfs_data/lmwang/Data/THUMOS14/25frame_val";
for file in /nfs_data/lmwang/Data/THUMOS14/val_video_org/*.mp4; do
    destination=${file/val_video_org/25frame_val};
#    mkdir -p "$destination";
    echo destination;
    ffmpeg -i "$file" -r 25  -strict -2 -y "$destination";
done

mkdir -p "/nfs_data/lmwang/Data/THUMOS14/25frame_test";
for file in /nfs_data/lmwang/Data/THUMOS14/test_video_org/*.mp4; do
    destination=${file/test_video_org/25frame_test};
#    mkdir -p "$destination";
    ffmpeg -i "$file" -r 25  -strict -2 -y "$destination";
done