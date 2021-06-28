import argparse, os, cv2
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from imageio import imread, imsave
from skimage.transform import resize
from path import Path
from tqdm import tqdm
import open3d as o3d

parser = argparse.ArgumentParser(description='Script for DispNet testing all val.txt sequences with corresponding GT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset_dir",    type=str,  default='data/outdoor')
parser.add_argument("--output_dir",     type=str,  default='outputs/outdoor/pretrained')
parser.add_argument("--weights",        type=str,  default="pretrained_models/r18_nyu/dispnet_model_best.pth.tar")
parser.add_argument('--resnet_layers',  type=int,  default=18, choices=[18, 50])
parser.add_argument("--model_img_height", type=int,  default=480)
parser.add_argument("--model_img_width",  type=int,  default=640)
parser.add_argument("--min_depth", default=1e-3)
parser.add_argument("--max_depth", default=80)                      #Use 10m for indoors and 80m for outdoors
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from torchvision import transforms
from models import DispResNet
from utils import tensor2array

def load_tensor_image(filename):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != args.model_img_height or w != args.model_img_width):
        img = resize(img, (args.model_img_height, args.model_img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img, np.transpose(img, (1, 2, 0))

def depth_pair_visualizer(pred, gt):
    inv_pred = 1 / (pred + 1e-6)
    inv_gt = 1 / (gt + 1e-6)
    vmax = np.percentile(inv_gt, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_gt.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)
    vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)
    return vis_pred, vis_gt

def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    if 'nyu' in args.dataset_dir or 'indoor' in args.dataset_dir:
        return abs_rel, log10, rmse, a1, a2, a3
    elif 'kitti' in args.dataset_dir or 'outdoor' in args.dataset_dir:
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate_depth(gt_depths, pred_depths):
      errors = []
      ratios = []
      resized_pred_depths = []
      print("==> Evaluating depth result...")
      for i in tqdm(range(len(pred_depths))):
          if pred_depths[i].mean() != -1:
              gt_depth = gt_depths[i]
              gt_height, gt_width = gt_depth.shape[:2]

              # resizing prediction (based on inverse depth)
              pred_inv_depth = 1 / (pred_depths[i] + 1e-6)
              pred_inv_depth = cv2.resize(pred_inv_depth, (gt_width, gt_height))
              pred_depth = 1 / (pred_inv_depth + 1e-6)
              
              mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)
              val_pred_depth = pred_depth[mask]
              val_gt_depth = gt_depth[mask]
              
              # median scaling is used for monocular evaluation
              ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
              ratios.append(ratio)
              val_pred_depth *= ratio
              resized_pred_depths.append(pred_depth * ratio)
              val_pred_depth[val_pred_depth < args.min_depth] = args.min_depth
              val_pred_depth[val_pred_depth > args.max_depth] = args.max_depth
              errors.append(compute_depth_errors(val_gt_depth, val_pred_depth))

      ratios = np.array(ratios)
      med = np.median(ratios)
      print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
      print(" Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(np.mean(ratios), np.std(ratios)))
      mean_errors = np.array(errors).mean(0)
      
      if 'nyu' in args.dataset_dir or 'indoor' in args.dataset_dir:
          print("\n  " + ("{:>8} | " * 6).format("abs_rel", "log10", "rmse", "a1", "a2", "a3"))
          print(("&{: 8.3f}  " * 6).format(*mean_errors.tolist()) + "\\\\")
      elif 'kitti' in args.dataset_dir or 'outdoor' in args.dataset_dir:
          print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
          print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
      return resized_pred_depths

@torch.no_grad()
def main():
    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.weights)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    
    all_pred_depths = []
    all_gt_depths = []
    val_folders = [line.strip() for line in open(dataset_dir/'val.txt')]
    for folder in val_folders:
        dst = output_dir/folder
        dst.makedirs_p()
        
        src = dataset_dir/folder
        test_files=sorted((src/'color/').files('*.jpg'))
        gt_files=sorted((src/'depth/').files('*.png'))
        intrinsics = np.genfromtxt(src/'cam.txt').astype(np.float32).reshape((3, 3))
        print('{} files to test'.format(len(test_files)))
        
        pred_depths = []
        gt_depths = []
        images = []
        for j in tqdm(range(len(test_files))):
            tensor_img, img = load_tensor_image(test_files[j])
            gt_depth = np.clip(imread(gt_files[j]).astype(np.float32), 0, args.max_depth * 1000)/1000
            output = disp_net(tensor_img)
            
            pred_disp = output.cpu().numpy()[0,0]
            pred_depth = 1/pred_disp
            #imsave(dst/'{:05d}_depth.png'.format(j), np.clip(65535.0 * pred_depth[:,:,np.newaxis]/100, 0.0, 65535.0).astype(np.uint16))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image((img/255.0).astype(np.float32)),
                                                                      o3d.geometry.Image(pred_depth))
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(args.model_img_width, args.model_img_height, intrinsics[0,0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]))
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            o3d.io.write_point_cloud(dst/"{:05}_cloud.pcd".format(j), pcd)
            
            pred_depths.append(pred_depth)
            gt_depths.append(gt_depth)
            images.append(img)
        
        pred_depths = evaluate_depth(gt_depths, pred_depths)
        for j in tqdm(range(len(images))):
            h = args.model_img_height
            w = args.model_img_width
            out_img = np.zeros((h, 3*w, 3))
            out_img[:, :w] = images[j]
            vis_pred, vis_gt = depth_pair_visualizer(pred_depths[j], gt_depths[j])
            out_img[:, w:2*w] = vis_pred
            out_img[:, 2*w:3*w] = vis_gt
            cv2.imwrite(dst/"{:05}.png".format(j), cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                
        all_pred_depths.extend(pred_depths)
        all_gt_depths.extend(gt_depths)
    _ = evaluate_depth(all_gt_depths, all_pred_depths)

if __name__ == '__main__':
    main()