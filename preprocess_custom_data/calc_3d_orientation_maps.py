import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
import trimesh


from gabor import filter_bank_gb3d as gabor3d


filename = './pipeline_ear_segmented_mesh/39_mesh.obj'

mesh = trimesh.load(filename)

# Voxelize the loaded mesh with a voxel size of 0.01. We also call hollow() to remove the inside voxels, which will help with color calculation
angel_voxel = mesh.voxelized(0.005)
voxel = angel_voxel.matrix

gabor_filters = np.asarray(gabor3d(plot=False))

voxel_torch = torch.from_numpy(voxel)[None, None, ...].type(torch.float32).to('cuda:0')
gabor_weights = torch.from_numpy(gabor_filters).unsqueeze(1).type(torch.float32).to('cuda:0')
F_orient = torch.abs(F.conv3d(voxel_torch, gabor_weights))
orientation_maps = F_orient.squeeze().argmax(0)
torch.save(orientation_maps, '39_orientation_map.pt')