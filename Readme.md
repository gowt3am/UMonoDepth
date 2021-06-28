# Unsupervised Scale-consistent Depth Learning from Video [Duplicate]
This Repository is a duplicate of [SC-SfMLearner-Release](https://github.com/JiawangBian/SC-SfMLearner-Release) with minor modifications.

## Installation and Setup
### Docker Environment:
Getting base image:
```shell script
$ docker pull nvcr.io/nvidia/pytorch:20.09-py3
```

Running base image:
```shell script
$ docker run --rm -it nvcr.io/nvidia/tensorflow:20.01-tf2-py3 bash
```

Installing dependencies:
```shell script
(docker)# apt update
(docker)# apt install -y ffmpeg libsm6 libxext6 libxrender-dev
(docker)# pip install -r requirements.txt
```

Additionally, to run ```view_trajectory.py```, build and install [Pangolin](https://github.com/stevenlovegrove/Pangolin) and its dependencies. Then build and install its Python bindings from [here](https://github.com/uoip/pangolin).

### Dataset Preparation:
Store datasets in ```./data/``` as

    dataset/train.txt
    dataset/val.txt
    dataset/scene_1/cam.txt
    dataset/scene_1/color/0000000.jpg
    dataset/scene_1/color/0000001.jpg
    ..
    dataset/scene_1/depth/0000000.png
    dataset/scene_1/depth/0000001.png
    ..
    dataset/scene_2/cam.txt
    dataset/scene_2/color/0000000.jpg
    ..
    dataset/scene_X/depth/XXXXXXX.png

1. ```train.txt``` contains the scene names to be used for training and ```val.txt``` contains the scene names to be used for testing and validation.
2. Each scene contains ```cam.txt``` with camera intrinsics stored as a 3x3 matrix.

### Pretrained Models:
Download models from the below links and store them under ```pretrained_models``` directory.

[ResNet18_KITTI](https://onedrive.live.com/?authkey=%21AP8Z6Tl8RC8waZo&id=36712431A95E7A25%212455&cid=36712431A95E7A25) | [ResNet50_KITTI](https://onedrive.live.com/?authkey=%21AP8Z6Tl8RC8waZo&id=36712431A95E7A25%212454&cid=36712431A95E7A25) | [ResNet18_NYU](https://onedrive.live.com/?authkey=%21AAnfSiMjlmnkizc&id=36712431A95E7A25%213206&cid=36712431A95E7A25) | [ResNet18_RectifiedNYU](https://onedrive.live.com/?authkey=%21AAnfSiMjlmnkizc&id=36712431A95E7A25%213207&cid=36712431A95E7A25)

1. Use NYU pretrained models for finetuning on Indoor scenes. Use KITTI pretrained models for training on Outdoor scenes.
2. For both scenes, training from scratch also seems to work well.

## Steps for running different components
1. Firstly, run the docker base image (with all installed dependencies).
 
### Training
1. Use ```train.sh``` to call ```train.py``` with required arguments.

### Testing
1. Use ```test.sh``` to call the 3 test scripts with required arguments.
2. ```test_depth.py``` will evaluate model predictions against ground truth depth after median scaling, for all validation sequences. It also generates 3D point clouds and visualization images of predictions.
3. ```infer_depth.py``` will generate depth maps for all validation sequences.
4. ```infer_pose.py``` will generate camera trajectories for all sequences provided in arguments.

### Viewing Trajectory
1. Ensure you have Pangolin (its Python binding) and PyOpenGL installed properly.
2. Run ```view_trajectory.py``` with proper arguments.