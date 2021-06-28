import argparse, time, csv, os
import numpy as np
from path import Path

parser = argparse.ArgumentParser(description='Training Script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data',                   type=str, default='data/outdoor')
parser.add_argument('--dataset',                type=str, default='outdoors', choices=['kitti', 'nyu', 'indoors', 'outdoors'])
parser.add_argument('--val_with_gt',            action='store_true')

parser.add_argument('--epochs',                 type=int,  default=50)
parser.add_argument('--batch_size',             type=int,  default=4)
parser.add_argument('--lr',                     type=float,default=1e-5)
parser.add_argument('--name',                   type=str, required=True)
parser.add_argument('--pretrained_disp',        type=str, default=None)
parser.add_argument('--pretrained_pose',        type=str, default=None)
parser.add_argument('--resnet_layers',          type=int, default=18, choices=[18, 50])
parser.add_argument('--print_freq',             type=int, default=10)
parser.add_argument('--gpu',                    type=str, default='0')

#############################################################################################################
parser.add_argument('--sequence_length',        type=int, default=3)
parser.add_argument('--folder_type',            type=str, default='sequence', choices=['sequence', 'pair'])
parser.add_argument('--workers',                type=int, default=4)
parser.add_argument('--padding_mode',           type=str, default='zeros', choices=['zeros', 'border'])
parser.add_argument('--num_scales',             type=int, default=1)
parser.add_argument('--epoch_size',             type=int,  default=0)
parser.add_argument('--momentum',               type=float,default=0.9)
parser.add_argument('--beta',                   type=float,default=0.999)
parser.add_argument('--weight_decay',           type=float,default=0)
parser.add_argument('-p', '--photo_loss_weight',          type=float, default=1.0)
parser.add_argument('-s', '--smooth_loss_weight',         type=float, default=0.1)
parser.add_argument('-c', '--geometry_consistency_weight',type=float, default=0.5)
parser.add_argument('--with_ssim',              type=bool, default=True)
parser.add_argument('--with_mask',              type=bool, default=True)
parser.add_argument('--with_auto_mask',         type=bool, default=True)
parser.add_argument('--with_pretrain',          type=bool, default=True)
parser.add_argument('--seed',                   type=int, default=0)
parser.add_argument('--log_output',             type=bool, default=True)
parser.add_argument('--log_summary',            default='summary.csv')
parser.add_argument('--log_full',               default='full.csv')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.backends.cudnn as cudnn
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
from tensorboardX import SummaryWriter

import models
import custom_transforms
from utils import tensor2array, save_checkpoint
from datasets import SequenceFolder, PairFolder, ValidationSet
from loss_functions import *
from logger import TermLogger, AverageMeter

best_error = -1
n_iter = 0
def main():
    global best_error, n_iter, device
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    train_transform = custom_transforms.Compose([custom_transforms.RandomHorizontalFlip(), custom_transforms.RandomScaleCrop(),
                                                 custom_transforms.ArrayToTensor(), normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(args.data, transform=train_transform, seed=args.seed, train=True,
                                   sequence_length=args.sequence_length, dataset=args.dataset)
    else:
        train_set = PairFolder(args.data, seed=args.seed, train=True, transform=train_transform)

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.val_with_gt:
        val_set = ValidationSet(args.data, transform=valid_transform, dataset=args.dataset)
    else:
        val_set = SequenceFolder(args.data, transform=valid_transform, seed=args.seed, train=False,
                                 sequence_length=args.sequence_length, dataset=args.dataset)

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    print("=> Creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

    if args.pretrained_disp:
        print("=> Using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
    if args.pretrained_pose:
        print("=> Using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    optim_params = [{'params': disp_net.parameters(), 'lr': args.lr}, {'params': pose_net.parameters(), 'lr': args.lr}]
    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)
    
    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])
    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])
    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()
    
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for 1 Epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.val_with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(args.save_path, {'epoch': epoch + 1, 'state_dict': disp_net.module.state_dict()}, {'epoch': epoch + 1, 'state_dict': pose_net.module.state_dict()}, is_best)
        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()
    pose_net.train()
    
    end = time.time()
    logger.train_bar.update(0)
    for i, files in enumerate(train_loader):
        tgt_img, ref_imgs, intrinsics, intrinsics_inv = files
        
        log_losses = i > 0 and n_iter % args.print_freq == 0
        data_time.update(time.time() - end)
        
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        loss_1, loss_3, valid_mask1, valid_mask2 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), args.batch_size)
        
        if log_losses:
            train_writer.add_scalar('train_photo_loss', loss_1.item(), n_iter)
            train_writer.add_scalar('train_smooth_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('train_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('train_total_loss', loss.item(), n_iter)
            train_writer.add_image('train Valid Mask1', valid_mask1[0].detach().cpu().numpy(), n_iter)
            #train_writer.add_image('train Valid Mask2', valid_mask2[0].detach().cpu().numpy(), n_iter)
                
        batch_time.update(time.time() - end)
        end = time.time()
        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break
        n_iter += 1
    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, files in enumerate(val_loader):
        tgt_img, ref_imgs, intrinsics, intrinsics_inv = files
        
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        loss_1, loss_3, _, __ = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        
        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
            output_writers[i].add_image('val Dispnet Output Normalized', tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'), epoch)
            output_writers[i].add_image('val Depth Output', tensor2array(tgt_depth[0][0], max_value=None), epoch)
        
        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()
        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['val_total_loss', 'val_photo_loss', 'val_smooth_loss', 'val_consistency_loss']


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    disp_net.eval()
    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)
        # check gt
        if depth.nelement() == 0:
            continue

        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]
        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image('val target Depth', tensor2array(depth_to_show, max_value=None), epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                output_writers[i].add_image('val target Disparity Normalized', tensor2array(disp_to_show, max_value=None, colormap='magma'), epoch)

            output_writers[i].add_image('val Dispnet Output Normalized', tensor2array(output_disp[0], max_value=None, colormap='magma'), epoch)
            output_writers[i].add_image('val Depth Output', tensor2array(output_depth[0], max_value=None), epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)
        errors.update(compute_errors(depth, output_depth, args.dataset))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names

def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]
    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)
    return tgt_depth, ref_depths

def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))
    return poses, poses_inv

if __name__ == '__main__':
    main()