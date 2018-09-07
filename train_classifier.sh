#!/usr/bin/env bash
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt21
#echo End1
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt22
#echo End2
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt23
#echo End3
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt24
#echo End4
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt25
#echo End5
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt26
#echo End6
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt27
#echo End7
#python train_classifier.py --batch-size=256 --epoch=50 --lr=8e-5 --fuse-type=average --model=mlp --exp-name=clf_avg_ucfandt28
#echo End8
python train_classifier.py --batch-size=256 --epoch=1 --fuse-type=none --model=mlp --exp-name=avg_score_unet --restore=models/simple_unet.pth.tar