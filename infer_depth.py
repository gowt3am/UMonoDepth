import argparse, os
import numpy as np
from imageio import imread, imsave
from skimage.transform import resize
from path import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Script for DispNet inference on val.txt sequences', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_dir",    type=str,  default='data/outdoor')
parser.add_argument("--output_dir",     type=str,  default='outputs/outdoor/pretrained') #'depths/NYU_FT_2'
parser.add_argument("--weights",        type=str,  default="pretrained_models/r18_nyu/dispnet_model_best.pth.tar")
parser.add_argument('--resnet_layers',  type=int,  default=18, choices=[18, 50])
parser.add_argument("--ratio",          type=float,default=20)
parser.add_argument("--model_img_height",     type=int,  default=480)
parser.add_argument("--model_img_width",      type=int,  default=640)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from models import DispResNet
from utils import tensor2array

def load_tensor_image(filename):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != args.model_img_height or w != args.model_img_height):
        img = resize(img, (args.model_img_height, args.model_img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img, np.transpose(img, (1, 2, 0))

@torch.no_grad()
def main():
    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.weights)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    
    val_folders = [line.strip() for line in open(dataset_dir/'val.txt')]
    for folder in val_folders:
        dst = output_dir/folder
        dst.makedirs_p()
        
        src = dataset_dir/folder
        test_files=sorted((src/'color/').files('*.jpg'))
        print('{} files to test'.format(len(test_files)))
        
        for j in tqdm(range(len(test_files))):
            tensor_img, img = load_tensor_image(test_files[j])
            output = disp_net(tensor_img)
            pred_disp = output.cpu().numpy()[0,0]
            pred_depth = 1/pred_disp
            imsave(dst/'{:05d}_depth.png'.format(j), np.clip(65535.0 * pred_depth[:,:,np.newaxis]/100, 0.0, 65535.0).astype(np.uint16))

if __name__ == '__main__':
    main()