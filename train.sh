#!/bin/bash
export PYTHONPATH='.'
python3 train.py --name outdoors_50allFT_noGT --dataset outdoors --data data/outdoor --epochs 50 --batch_size 4 --lr 1e-5 --print_freq 100 --gpu 0 --pretrained_disp models/resnet18_depth_256/dispnet_model_best.pth.tar --pretrained_pose models/resnet18_depth_256/exp_pose_model_best.pth.tar &\
python3 train.py --name outdoors_50allFT --dataset outdoors --data data/outdoor --val_with_gt --epochs 50 --batch_size 4 --lr 1e-5 --print_freq 100 --gpu 1 --pretrained_disp models/resnet18_depth_256/dispnet_model_best.pth.tar --pretrained_pose models/resnet18_depth_256/exp_pose_model_best.pth.tar &\
python3 train.py --name outdoors_50allScratch --dataset outdoors --data data/outdoor --val_with_gt --epochs 50 --batch_size 4 --lr 1e-5 --print_freq 100 --gpu 1 &\