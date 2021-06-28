#!/bin/bash
export PYTHONPATH='.'
python3 test_depth.py --weights checkpoints/outdoors_50/dispnet_model_best.pth.tar --output_dir outputs/outdoor/50all &&\
python3 infer_pose.py --weights checkpoints/outdoors_50/exp_pose_model_best.pth.tar --output_dir outputs/outdoor/50all &&\
python3 infer_depth.py --weights checkpoints/outdoors_50/dispnet_model_best.pth.tar --output_dir outputs/outdoor/50all