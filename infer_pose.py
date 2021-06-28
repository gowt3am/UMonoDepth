import argparse, os
import numpy as np
from imageio import imread
from skimage.transform import resize
from path import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Script for PoseNet inference on args.test_seq sequences', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_dir",    type=str,  default='data/outdoor')
parser.add_argument("--output_dir",     type=str,  default='outputs/outdoor/pretrained')
parser.add_argument("--weights",        type=str,  default="pretrained_models/r18_nyu/exp_pose_model_best.pth.tar")
parser.add_argument("--test_seq",       type=str,  default=['run1', 'run2', 'run3'])
parser.add_argument("--model_img_height",     type=int,  default=480)
parser.add_argument("--model_img_width",      type=int,  default=640)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from models import PoseResNet
from inverse_warp import pose_vec2mat

def load_tensor_image(filename):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != args.model_img_height or w != args.model_img_width):
        img = resize(img, (args.model_img_height, args.model_img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img, np.transpose(img, (1, 2, 0))

@torch.no_grad()
def main():
    pose_net = PoseResNet(18, False).to(device)
    weights = torch.load(args.weights)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    
    for folder in args.test_seq:
        src = dataset_dir/folder
        test_files=sorted((src/'color/').files('*.jpg'))
        n = len(test_files)
        print('{} is length of sequence'.format(n))
      
        global_pose = np.eye(4)
        poses = [global_pose[0:3, :].reshape(1, 12)]

        for i in tqdm(range(n-1)):
            tensor_img1, _ = load_tensor_image(test_files[i])
            tensor_img2, _ = load_tensor_image(test_files[i+1])
            pose = pose_net(tensor_img1, tensor_img2)
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)
            
            poses.append(global_pose[0:3, :].reshape(1, 12))
        poses = np.concatenate(poses, axis=0)
        np.savetxt(output_dir/'{}_trajectory.txt'.format(folder), poses, delimiter=' ', fmt='%1.8e')

if __name__ == '__main__':
    main()