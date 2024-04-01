import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import os
import json
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
from pathlib import Path
import toml
import math
from .cameras import OptimizableCameras
from pytorch3d.io import load_obj
import pickle

import sys
sys.path.append(os.path.join(sys.path[0], '../..'))
from NeuS.models.dataset import load_K_Rt_from_P
from src.utils.util import glob_imgs, tensor2image


class H3DSDatasetTorch(Dataset):
    def __init__(
        self,
        ray_batch_size: int,
        conf: dict,
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda')
        self.ray_batch_size = ray_batch_size
        self.conf = conf

        self.data_dir = Path(conf.get('data_dir'))
        self.render_cameras_name = conf.get('render_cameras_name')
        self.object_cameras_name = conf.get('object_cameras_name')


        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'image')))


        self.orientations_np, self.hair_masks_np, self.hair_masks_np_white_gray = None, None, None
        self.num_bins = conf.get('orient_num_bins')  

        fitted_camera_path = conf.get('fitted_camera_path', '')
        self.fitted_camera_path = fitted_camera_path

        self.masks_lis  = sorted(glob_imgs(os.path.join(self.data_dir, 'mask')))

        self.hair_masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'hair_mask')))
                
        self.orientations_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'orientation_maps')))
        self.variance_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'confidence_maps')))    
        
        self.scale_mats_np_0 = camera_dict['scale_mat_0'].astype(np.float32)

        self.n_images = len(self.images_lis)
        self.filter_views()

    def filter_views(self):
        print('Filter scene!')
        with open(self.views_idx, 'rb') as f:
            filter_idx = pickle.load(f)
            print(filter_idx)
        self.cameras = [self.cameras[i] for i in filter_idx]
        self.hair_masks_lis = [self.hair_masks_lis[i] for i in filter_idx]
        self.orientations_lis = [self.orientations_lis[i] for i in filter_idx]
        self.variance_lis = [self.variance_lis[i] for i in filter_idx]
        self.images_lis = [self.images_lis[i] for i in filter_idx]
        self.masks_lis = [self.masks_lis[i] for i in filter_idx]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx: int):
        image_np = cv.imread(self.images_lis[idx]) / 255.0
        mask_np = cv.imread(self.masks_lis[idx]) / 255.0
        hair_mask_np = cv.imread(self.hair_masks_lis[idx]) / 255.0
        orientations_np = cv.imread(self.orientations_lis[idx]) / float(self.num_bins) * math.pi
        variance_np = np.load(self.variance_lis[idx])

        world_mat_np = self.camera_dict['world_mat_%d' % idx].astype(np.float32)
        scale_mat_np = self.camera_dict['scale_mat_%d' % idx].astype(np.float32)

        P = world_mat_np @ scale_mat_np
        P = P[:3, :4]
        intrinsics_np, pose_np = load_K_Rt_from_P(None, P)
        intrinsic = torch.from_numpy(intrinsics_np).float()
        pose = torch.from_numpy(pose_np).float()

        image = torch.from_numpy(image_np.astype(np.float32)).cpu()  # [H, W, 3]
        mask  = torch.from_numpy(mask_np.astype(np.float32)).cpu()   # [H, W, 3]
        hair_mask = torch.from_numpy(hair_mask_np.astype(np.float32)).cpu()
        orientation_map = torch.from_numpy(orientations_np.astype(np.float32)).cpu()
        variance_map = torch.from_numpy(variance_np.astype(np.float32)).cpu()[..., None]
        confidence_map = 1 / variance_map ** 2

        intrinsic = intrinsic.to(self.device)   # [4, 4]
        pose = pose.to(self.device)  # [4, 4]

        print(f'Fitted camera path is set to {self.fitted_camera_path}')
        if self.fitted_camera_path:
            camera_model = OptimizableCameras(len(self.images_lis), pretrain_path=self.fitted_camera_path).to(self.device)
            with torch.no_grad():
                print('camera model create')
                intrinsic, pose = camera_model(idx, intrinsic, pose)

        intrinsic_inv = torch.inverse(intrinsic)
        focal = intrinsic[0, 0]
        pose_inv = torch.inverse(pose)
        H, W = image.shape[0], image.shape[1]
        image_pixels = H * W

        if self.conf.get('mask_based_ray_sampling', False):
            binary_mask = mask[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)
            binary_hair_mask = hair_mask[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        if 'scale_mat_0' in self.camera_dict.keys():
            # Object scale mat: region of interest to **extract mesh**
            object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
            object_bbox_min = np.linalg.inv(self.scale_mats_np_0) @ object_scale_mat @ object_bbox_min[:, None]
            object_bbox_max = np.linalg.inv(self.scale_mats_np_0) @ object_scale_mat @ object_bbox_max[:, None]
            object_bbox_min = object_bbox_min[:, 0]
            object_bbox_max = object_bbox_max[:, 0]

        object_bbox_min_final = object_bbox_min[:3]
        object_bbox_max_final = object_bbox_max[:3]

        radii = torch.empty(W, H)
        tx = torch.linspace(0, W - 1, W)
        ty = torch.linspace(0, H - 1, H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        v = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None].cuda()).squeeze()  # W, H, 3
            
        ###
        ### Code below is borrowed from https://github.com/hjxwhy/mipnerf_pl/blob/master/datasets/datasets.py
        ###
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(torch.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx * 2 / math.sqrt(12)
        radii_final = radii.transpose(0, 1)[..., None]

        if self.conf.get('mask_based_ray_sampling', False):
            # Sample 50% hair rays, 45% foreground rays and 5% background rays
            num_pixels_bg = round(self.ray_batch_size * 0.05)
            num_pixels_fg = round(self.ray_batch_size * 0.45)
            num_pixels_hair = self.ray_batch_size - num_pixels_bg - num_pixels_fg

            pixels_bg = torch.nonzero(~binary_mask)
            pixels_bg = pixels_bg[torch.randperm(pixels_bg.shape[0])][:num_pixels_bg]

            pixels_fg = torch.nonzero(binary_mask)
            pixels_fg = pixels_fg[torch.randperm(pixels_fg.shape[0])][:num_pixels_fg]

            pixels_hair = torch.nonzero(binary_hair_mask)
            pixels_hair = pixels_hair[torch.randperm(pixels_hair.shape[0])][:num_pixels_hair]

            pixels_x, pixels_y = torch.cat([pixels_bg, pixels_fg, pixels_hair], dim=0).cuda().split(1, dim=1)
            pixels_x = pixels_x[:, 0]
            pixels_y = pixels_y[:, 0]
        else:
            pixels_x = torch.randint(low=0, high=W, size=[self.ray_batch_size]).cpu()
            pixels_y = torch.randint(low=0, high=H, size=[self.ray_batch_size]).cpu()

        color = image[(pixels_y, pixels_x)]    # batch_size, 3
        mask = mask[(pixels_y, pixels_x)]      # batch_size, 3
        radii_final = radii_final[(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
            
        hair_mask = hair_mask[(pixels_y, pixels_x)]
        orientation = orientation_map[(pixels_y, pixels_x)]
        confidence = confidence_map[(pixels_y, pixels_x)] 

        concated_tensor = torch.cat([p.cpu(),
                            radii_final.cpu(), 
                            color, 
                            mask[:, :1],
                            hair_mask[:, :1]], dim=-1)
        

        orient = cv.imread(self.orientations_lis[idx])[..., :1] / float(self.num_bins) * math.pi
        cos = np.cos(orient)
        orient_img = np.concatenate([cos * (cos >= 0), np.sin(orient), np.abs(cos) * (cos < 0)], axis=-1)
        orient_img = (orient_img * 255).round().astype('uint8')
        orient_img_final = cv.resize(orient_img, (W // 1, H // 1), interpolation=cv.INTER_CUBIC).clip(0, 255)

        return_dict = {
            'concatenated_tensor': torch.cat([
                concated_tensor,
                orientation[:, :1], 
                confidence[:, :1]], dim=-1
            ).cuda(),
            'intrinsic': intrinsic,
            'pose': pose,
            'orient_image': orient_img_final,
            'image': image.permute(2, 0, 1).cuda(),
            'hair_mask': hair_mask.permute(2, 0, 1).cuda(),

        }
        return return_dict


class MonocularDatasetTorch(Dataset):
    def __init__(
            self,
            ray_batch_size: int,
            conf: dict,
        ) -> None:
        super().__init__()
        self.device = torch.device('cuda')
        self.ray_batch_size = ray_batch_size
        self.conf = conf

        self.data_dir = Path(conf.get('data_dir'))
        self.render_cameras_name = conf.get('render_cameras_name')
        self.object_cameras_name = conf.get('object_cameras_name')

        self.views_idx = conf.get('views_idx', '')

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        
        fitted_camera_path = conf.get('fitted_camera_path')
        self.fitted_camera_path = fitted_camera_path

        # Define scale into unit sphere
        self.scale_mat = np.eye(4, dtype=np.float32)
        if conf.get('path_to_scale', None) is not None:
            with open(conf['path_to_scale'], 'rb') as f:
                transform = pickle.load(f)
                print('upload transform', transform, conf['path_to_scale'])
                self.scale_mat[:3, :3] *= transform['scale']
                self.scale_mat[:3, 3] = np.array(transform['translation'])
    
        self.num_bins = conf.get('orient_num_bins')
        self.intrinsics_all = []
        self.pose_all = []

        self.images_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'image')))

        self.masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'mask')))
#        load hair mask
        self.hair_masks_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'hair_mask')))
#             load orientations
        self.orientations_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'orientation_maps')))
#            load variance
        self.variance_lis = sorted(glob_imgs(os.path.join(self.data_dir, 'confidence_maps')))
#         Load camera
        self.cameras = camera_dict['arr_0']
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        self.n_images = len(self.images_lis)
        if self.views_idx:
            self.filter_views()

        print("Number of views:", self.n_images) 
        for i in range(self.n_images):
            print(f"Iteration {i}")
            world_mat = self.cameras[i]  
            P = world_mat @ self.scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)                       
            self.pose_all.append(torch.from_numpy(pose).float())
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())    
        
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        
        if fitted_camera_path:
            camera_model = OptimizableCameras(len(self.images_lis), pretrain_path=fitted_camera_path).to(self.device)
            with torch.no_grad():
                print('camera model create')
                self.intrinsics_all, self.pose_all = camera_model(torch.arange(len(self.images_lis)), self.intrinsics_all, self.pose_all)

    def filter_views(self):
        print('Filter scene!')
        with open(self.views_idx, 'rb') as f:
            filter_idx = pickle.load(f)
            print(filter_idx)
        self.cameras = [self.cameras[i] for i in filter_idx]
        self.hair_masks_lis = [self.hair_masks_lis[i] for i in filter_idx]
        self.orientations_lis = [self.orientations_lis[i] for i in filter_idx]
        self.variance_lis = [self.variance_lis[i] for i in filter_idx]
        self.images_lis = [self.images_lis[i] for i in filter_idx]
        self.masks_lis = [self.masks_lis[i] for i in filter_idx]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx: int):

        image_np = cv.imread(self.images_lis[idx]) / 255.0
        mask_np = cv.imread(self.masks_lis[idx]) / 255.0
        hair_mask_np = cv.imread(self.hair_masks_lis[idx]) / 255.0
        orientations_np = cv.imread(self.orientations_lis[idx]) / float(self.num_bins) * math.pi
        variance_np = np.load(self.variance_lis[idx])

        image = torch.from_numpy(image_np.astype(np.float32)).cpu()  # [H, W, 3]
        mask  = torch.from_numpy(mask_np.astype(np.float32)).cpu()   # [H, W, 3]
        hair_mask = torch.from_numpy(hair_mask_np.astype(np.float32)).cpu() # [H, W, 3]
        orientation_map = torch.from_numpy(orientations_np.astype(np.float32)).cpu()
        variance_map = torch.from_numpy(variance_np.astype(np.float32)).cpu()[..., None]
        confidence_map = 1 / variance_map ** 2
        confidence_map[torch.isinf(confidence_map)] = 100000

        intrinsic = self.intrinsics_all[idx]
        pose = self.pose_all[idx]

        intrinsic_inv = torch.inverse(intrinsic) # [1, 4, 4]
        focal = intrinsic[0, 0]
        pose_inv = torch.inverse(pose)
        H, W = image.shape[0], image.shape[1]
        image_pixels = H * W

        if self.conf.get('mask_based_ray_sampling', False):
            binary_mask = mask[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)
            binary_hair_mask = hair_mask[..., 0] > self.conf.get('mask_binarization_threshold', 0.5)


        radii = torch.empty(W, H)
        tx = torch.linspace(0, W - 1, W)
        ty = torch.linspace(0, H - 1, H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        v = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None].cuda()).squeeze()  # W, H, 3
            
        ###
        ### Code below is borrowed from https://github.com/hjxwhy/mipnerf_pl/blob/master/datasets/datasets.py
        ###
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(torch.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx * 2 / math.sqrt(12)
        radii_final = radii.transpose(0, 1)[..., None]

        if self.conf.get('mask_based_ray_sampling', False):
            # Sample 50% hair rays, 45% foreground rays and 5% background rays
            num_pixels_bg = round(self.ray_batch_size * 0.05)
            num_pixels_fg = round(self.ray_batch_size * 0.45)
            num_pixels_hair = self.ray_batch_size - num_pixels_bg - num_pixels_fg

            pixels_bg = torch.nonzero(~binary_mask)
            pixels_bg = pixels_bg[torch.randperm(pixels_bg.shape[0])][:num_pixels_bg]

            pixels_fg = torch.nonzero(binary_mask)
            pixels_fg = pixels_fg[torch.randperm(pixels_fg.shape[0])][:num_pixels_fg]

            pixels_hair = torch.nonzero(binary_hair_mask)
            pixels_hair = pixels_hair[torch.randperm(pixels_hair.shape[0])][:num_pixels_hair]

            pixels_x, pixels_y = torch.cat([pixels_bg, pixels_fg, pixels_hair], dim=0).cuda().split(1, dim=1)
            pixels_x = pixels_x[:, 0]
            pixels_y = pixels_y[:, 0]
        else:
            pixels_x = torch.randint(low=0, high=W, size=[self.ray_batch_size]).cpu()
            pixels_y = torch.randint(low=0, high=H, size=[self.ray_batch_size]).cpu()

        color = image[(pixels_y, pixels_x)]    # batch_size, 3
        mask_part = mask[(pixels_y, pixels_x)]      # batch_size, 3
        radii_final = radii_final[(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
            
        hair_mask_part = hair_mask[(pixels_y, pixels_x)]
        orientation_part = orientation_map[(pixels_y, pixels_x)]
        confidence_part = confidence_map[(pixels_y, pixels_x)] 

        concated_tensor = torch.cat([p.cpu(),
                            radii_final.cpu(), 
                            color, 
                            mask_part[:, :1],
                            hair_mask_part[:, :1]], dim=-1)
        

        orient = cv.imread(self.orientations_lis[idx])[..., :1] / float(self.num_bins) * math.pi
        cos = np.cos(orient)
        orient_img = np.concatenate([cos * (cos >= 0), np.sin(orient), np.abs(cos) * (cos < 0)], axis=-1)
        orient_img = (orient_img * 255).round().astype('uint8')
        orient_img_final = cv.resize(orient_img, (W // 1, H // 1), interpolation=cv.INTER_CUBIC).clip(0, 255)

        return_dict = {
            'concatenated_tensor': torch.cat([
                concated_tensor,
                orientation_part[:, :1], 
                confidence_part[:, :1]], dim=-1
            ).cuda(),
            'intrinsic': intrinsic,
            'pose': pose,
            'orient_image': orient_img_final,
            'image': image.permute(2, 0, 1).cuda(),
            'hair_mask': hair_mask.permute(2, 0, 1).cuda(),
            'H': H,
        }
        return return_dict