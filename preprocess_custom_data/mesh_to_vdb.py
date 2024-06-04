import pyopenvdb as vdb
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import pytorch3d
from pytorch3d.io import load_obj


from gabor import filter_bank_gb3d as gabor3d


filename = './implicit-hair-data/data/nphm/039/39_mesh_orig.obj'
mesh = trimesh.load(filename)
angel_voxel = mesh.voxelized(0.005)
voxel = angel_voxel.matrix.astype(float)
print(f'voxel size is {voxel.shape}')


grid = vdb.FloatGrid()
grid.copyFromArray(voxel)
grid.activeVoxelCount() == voxel.size
vdb.write('./implicit-hair-data/data/nphm/039/nphm_39_sparse.vdb', grids=[grid])