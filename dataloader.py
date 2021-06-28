import random
import numpy as np
from path import Path
from imageio import imread
import torch
import torch.utils.data as data

def load_as_float(path):
    return imread(path).astype(np.float32)

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/color/0000000.jpg
        root/scene_1/color/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/color/0000000.jpg
        .
        transform functions must take in a list of images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        
        for scene in self.scenes:
            print(scene)
            imgs = sorted((scene/'color/').files('*.jpg'))
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

##############################################################################

class PairFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000_0.jpg
        root/scene_1/0000001_1.jpg
        ..
        root/scene_1/cam.txt
        .
        transform functions must take in a list of images and a numpy array (usually intrinsics matrix)
    """
    def __init__(self, root, seed=None, train=True, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self,):
        pair_set = []
        for scene in self.scenes:
            imgs = sorted(scene.files('*.jpg'))
            intrinsics = sorted(scene.files('*.txt'))
            #intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            for i in range(0, len(imgs)-1, 2):
                intrinsic = np.genfromtxt(intrinsics[int(i/2)]).astype(np.float32).reshape((3, 3))
                sample = {'intrinsics': intrinsic, 'tgt': imgs[i], 'ref_imgs': [imgs[i+1]]}
                pair_set.append(sample)
        
        random.shuffle(pair_set)
        self.samples = pair_set
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

##############################################################################

class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/color/0000000.jpg
        root/scene_1/depth/0000000.png
        root/scene_1/color/0000001.jpg
        root/scene_1/depth/0000001.png
        ..
        root/scene_2/color/0000000.jpg
        root/scene_2/depth/0000000.png
        .
        transform functions must take in a list of images and a numpy array which can be None
    """
    def __init__(self, root, transform=None, dataset='nyu'):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depth = crawl_folders(self.scenes, self.dataset)
        
    def crawl_folders(folders_list, dataset='nyu'):
        imgs = []
        depths = []
        for folder in folders_list:
            if dataset in ['indoors', 'outdoors']:
                current_imgs = sorted((folder/'color/').files('*.jpg'))
                current_depth = sorted((folder/'depth/').files('*.png'))
            else:
                current_imgs = sorted(folder.files('*.jpg'))
                if dataset == 'nyu':
                    current_depth = sorted((folder/'depth/').files('*.png'))
                elif dataset == 'kitti':
                    current_depth = sorted(folder.files('*.npy'))
            imgs.extend(current_imgs)
            depths.extend(current_depth)
        return imgs, depths
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)
        if self.dataset=='nyu':
            depth = torch.from_numpy(imread(self.depth[index]).astype(np.float32)).float()/5000
        elif self.dataset=='kitti':
            depth = torch.from_numpy(np.load(self.depth[index]).astype(np.float32))
        elif self.dataset=='indoors':
            depth = torch.from_numpy(np.clip(imread(self.depth[index]).astype(np.float32), 0, 10000)).float()/1000
        elif self.dataset=='outdoors':
            depth = torch.from_numpy(np.clip(imread(self.depth[index]).astype(np.float32), 0, 80000)).float()/1000

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth