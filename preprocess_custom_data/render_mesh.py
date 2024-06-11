import os

import torch
import pytorch3d.loss
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.ops import norm_laplacian
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


def make_renderer(R, T, device='cuda:0'):
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=(512, 512), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    # lights = PointLights(device=device_render, location=[[0.0, -3.0, -5.0]])
    # lights = PointLights(device=device_render, location=[[0.0, 100, 100]])
    lights = cameras.get_camera_center()
    lights = PointLights(device=device, location=lights)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        # shader=HardFlatShader(device=device_render, cameras=cameras, lights=lights)
    )
    return renderer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mesh = IO().load_mesh('implicit-hair-data/data/nphm/039/39_mesh_orig.obj', device=device)
verts_rgb = torch.ones_like(mesh.verts_packed())[None]
textures = TexturesVertex(verts_features=verts_rgb.to(device))
mesh.textures = textures
template_center = mesh.get_bounding_boxes().mean(dim=-1)   # replace with the actual mesh center
template_center_1 = mesh.verts_packed().mean(dim=0, keepdim=True)

renderers = [
    make_renderer(*look_at_view_transform(0.9, 180, 90, at=template_center)),    # left
    make_renderer(*look_at_view_transform(0.9, 180, 180, at=template_center)),   # frontal
    make_renderer(*look_at_view_transform(0.9, 0, 90, at=template_center)),      # right
    make_renderer(*look_at_view_transform(1.0, 90, 0, at=template_center)),      # above
    make_renderer(*look_at_view_transform(1.0, 180, 0, at=template_center)),      # back
    make_renderer(*look_at_view_transform(0.9, 0, 90, at=template_center)),      # right
]

for i, R in enumerate(renderers):
    images = R(mesh)[0, ..., :3].cpu().data.numpy()

    # plt.figure(figsize=(10, 5))
    plt.imshow(images)
    plt.axis("off")
    # plt.title(f'iteration: {step_no} / {n_iter}')
    plt.tight_layout()
    # plt.title(f'iteration: {step_no} / {n_train_steps}; loss_lapl = {loss_dict["laplacian_term"]}')
    save_dir = './'
    plt.savefig(os.path.join(save_dir, f'rendered_mesh_{i}.jpg'), dpi=300)