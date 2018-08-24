#!/bin/bash

set -e

for file in /nfs_data/lmwang/Data/t15background/flow/*; do
    destination=${file/flow/rgb};
    mkdir -p "$destination";
#    echo "$file"/img*;
    mv "$file"/img* "$destination";
done