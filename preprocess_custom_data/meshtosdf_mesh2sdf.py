import os
import sys
import trimesh
import mesh2sdf
import numpy as np
import time


filename = '/home/lazuardi/NeuralHaircutNPHM/implicit-hair-data/data/nphm/039/39_mesh.obj'

mesh_scale = 0.8
size = 256
level = 2 / size

mesh = trimesh.load(filename, force='mesh')

# normalize mesh
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale

# fix mesh
print(f'Computing SDF with resolution {size}...')
t0 = time.time()
sdf, mesh = mesh2sdf.compute(
    vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
t1 = time.time()

# output
np.save(filename[:-8] + '.npy', sdf)
mesh.vertices = mesh.vertices / scale + center
# mesh.export('mesh_256.obj')