import math
import gc


import matplotlib.pyplot as plt
import numpy as np
import pytorch3d.structures
import torch
import torch.nn.functional as F
from torch import nn
import trimesh

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import cm


class Conv3dGabor(nn.Module):
    '''
    Applies a 3d convolution over an input signal using Gabor filter banks.
    WARNING: the size of the kernel must be an odd number otherwise it'll be shifted with respect to the origin
    Refer to https://github.com/m-evdokimov/pytorch-gabor3d
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 size: int,                
                 sigma=3,
                 gamma_y=0.5,
                 gamma_z=0.5,
                 lambd=6,
                 psi=0.,
                 padding=None,
                 device='cuda'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = in_channels * out_channels
        self.size = size
        self.device = device

        if padding:
            self.padding = padding
        else:
            self.padding = 0

        # all additional axes are made for correct broadcast
        # the bounds of uniform distribution adjust manually for every size (rn they're adjusted for 5x5x5 filters)
        # for better understanding: https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97

        self.sigma = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * sigma

        self.gamma_y = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * gamma_y
        self.gamma_z = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * gamma_z

        self.lambd = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * lambd

        self.psi = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * psi

        self.angles = torch.zeros(size=(self.num_filters, 3)).to(self.device)
        num_angles_per_axis = round(math.sqrt(self.num_filters))
        angle_step = math.pi / num_angles_per_axis
        # use polar coordinate, theta round with x, phi round with y
        for i_theta in range(num_angles_per_axis):
            for j_phi in range(num_angles_per_axis):
                rot_angle = torch.tensor([0, j_phi * angle_step, i_theta * angle_step]).to(self.device)
                self.angles[i_theta * num_angles_per_axis + j_phi] = rot_angle

        self.kernels = self.init_kernel()

    def init_kernel(self):
        '''
        Initialize a gabor kernel with given parameters
        Returns torch.Tensor with size (out_channels, in_channels, size, size, size)
        '''
        lambd = self.lambd
        psi = self.psi

        sigma_x = self.sigma
        sigma_y = self.sigma * self.gamma_y
        sigma_z = self.sigma * self.gamma_z
        R = self.get_rotation_matrix().reshape(self.num_filters, 3, 3, 1, 1, 1)

        c_max, c_min = int(self.size / 2), -int(self.size / 2)
        (x, y, z) = torch.meshgrid(torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1), indexing='ij') # for future warning

        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        # meshgrid for every filter
        x = x.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)
        y = y.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)
        z = z.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)

        x_prime = z * R[:, 2, 0] + y * R[:, 2, 1] + x * R[:, 2, 2]
        y_prime = z * R[:, 1, 0] + y * R[:, 1, 1] + x * R[:, 1, 2]
        z_prime = z * R[:, 0, 0] + y * R[:, 0, 1] + x * R[:, 0, 2]

        yz_prime = torch.sqrt(y_prime ** 2 + z_prime ** 2)

        # gabor formula
        kernel = torch.exp(-.5 * (x_prime ** 2 / sigma_x ** 2 + y_prime ** 2 / sigma_y ** 2 + z_prime ** 2 / sigma_z ** 2)) \
                 * torch.cos(2 * math.pi * yz_prime / (lambd + 1e-6) + psi)

        return kernel.reshape(self.out_channels, self.in_channels, self.size, self.size, self.size).contiguous()

    def get_rotation_matrix(self):
        '''
        Makes 3d rotation matrix.
        R_x = torch.Tensor([[cos_a, -sin_a, 0],
                           [sin_a, cos_a,  0],
                           [0,     0,      1]],)
        R_y = torch.Tensor([[cos_b,  0, sin_b],
                           [0    ,  1,    0],
                           [-sin_b, 0, cos_b]])
        R_z = torch.Tensor([[1,  0,     0],
                           [0,  cos_g, -sin_g],
                           [0,  sin_g, cos_g]])
        '''

        sin_a, cos_a = torch.sin(self.angles[:, 0]), torch.cos(self.angles[:, 0])
        sin_b, cos_b = torch.sin(self.angles[:, 1]), torch.cos(self.angles[:, 1])
        sin_g, cos_g = torch.sin(self.angles[:, 2]), torch.cos(self.angles[:, 2])

        R_x = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_x[:, 0, 0] = cos_a
        R_x[:, 0, 1] = -sin_a
        R_x[:, 1, 0] = sin_a
        R_x[:, 1, 1] = cos_a
        R_x[:, 2, 2] = 1

        R_y = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_y[:, 0, 0] = cos_b
        R_y[:, 0, 2] = sin_b
        R_y[:, 2, 0] = -sin_b
        R_y[:, 2, 2] = cos_b
        R_y[:, 1, 1] = 1

        R_z = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_z[:, 1, 1] = cos_g
        R_z[:, 1, 2] = -sin_g
        R_z[:, 2, 1] = sin_g
        R_z[:, 2, 2] = cos_g
        R_z[:, 0, 0] = 1

        return R_x @ R_y @ R_z

    def forward(self, x):
        with torch.no_grad():
            x = F.conv3d(x, weight=self.kernels, padding=self.padding)
        return x


def plot_3d_voxels(data, transparency_threshold=0, skew_factor=1.2, sampling_ratio=0.3, multi_view=True, colormap='magma',save_img=False ,channel_first=False):

    """
    This function visualizes 3D data (voxels) with one channel using a 3D scatter plot.

    Parameters:
    data (numpy.ndarray): A 4D numpy array representing the voxel grid.

    transparency_threshold (float): A threshold below which voxel values are considered transparent.

    skew_factor (float): A factor by which the voxel values are skewed before visualizing.
                         This puts emphasis on higher values and suppresses lower ones.
                         skewed_values = voxel_values ** skew_factor

    sampling_ratio (float): The ratio of non-zero voxels to sample and plot. Must be between 0 and 1.

    multi_view (boolean): Plot different views (perspective, top, bottom, front, left, right)

    colormap (string): The colormap of choice

    channel_first (boolean): Is the first dimension of your data the channel?

    save_img (boolean): Saves the plots as PDF and PNG. Change the path and file name to your liking.

    """



    # Check if the input data is a 4D numpy array
    if not isinstance(data, np.ndarray) or data.ndim != 4:
        raise ValueError("Input data must be a 4D numpy array")

    # Check if the transparency_threshold and skew_factor are valid numbers
    if not isinstance(transparency_threshold, (int, float)):
        raise ValueError("Transparency threshold must be a number")
    if not isinstance(skew_factor, (int, float)):
        raise ValueError("Skew factor must be a number")

    # Check if the sampling_ratio is a valid number between 0 and 1
    if not isinstance(sampling_ratio, (int, float)) or sampling_ratio < 0 or sampling_ratio > 1:
        raise ValueError("Sampling ratio must be a number between 0 and 1")

     # Reshape the data
    if channel_first==True:
        pass
    else:
        data = data.reshape((1, data.shape[0], data.shape[1], data.shape[2]))


    # Number of slices (z-dimension in the 3D space)
    num_slices = data.shape[-1]

    # Create figure and grid
    fig = plt.figure(figsize=(10, num_slices * 2))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1] * 5, width_ratios=[1,3,1])
    ax_main = plt.subplot(gs[:, 1], projection='3d')

    # Create colormap and calculate color and transparency for non-zero voxels
    magma_cmap = cm.get_cmap(colormap)
    voxel_indices = np.argwhere(data > 0)
    voxel_values = data[data > 0] / np.max(data)

    # Random sampling of non-zero points
    sampled_indices = np.random.choice(np.arange(voxel_indices.shape[0]), size=int(voxel_indices.shape[0] * sampling_ratio), replace=False)
    voxel_indices = voxel_indices[sampled_indices]
    voxel_values = voxel_values[sampled_indices]

    #Skew the values for visualisatin
    skewed_values = voxel_values ** skew_factor

    colors = magma_cmap(skewed_values)
    colors[:, 3] = np.where(voxel_values > transparency_threshold, voxel_values, 0)

    # Plot the main 3D scatter plot
    ax_main.scatter(voxel_indices[:, 1], voxel_indices[:, 2], voxel_indices[:, 3], c=colors, marker='s')

    if multi_view==True:
        #Diffrent views
        views = [
            {'title': 'Perspective View', 'elev': 30, 'azim': -45},
            {'title': 'Top View', 'elev': 90, 'azim': -95},
            {'title': 'Bottom View', 'elev': -90, 'azim': -95},
            {'title': 'Front View', 'elev': 5, 'azim': -85},
            {'title': 'Right View', 'elev': 5, 'azim': -20},
            {'title': 'Left View', 'elev': 5, 'azim': -160}
        ]

        # Plot the 3D different views scatter plot
        fig = plt.figure(figsize=(12, 12))
        for i, view in enumerate(views, start=1):
            ax = fig.add_subplot(2, 3, i, projection='3d')
            ax.scatter(voxel_indices[:, 1], voxel_indices[:, 2], voxel_indices[:, 3], c=colors, marker='s')
            ax.view_init(elev=view['elev'], azim=view['azim'])
            ax.set_title(view['title'])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(False)
            ax.xaxis.pane.set_edgecolor('k')
            ax.yaxis.pane.set_edgecolor('k')
            ax.zaxis.pane.set_edgecolor('k')

            ax.xaxis.pane.set_linewidth(2)
            ax.yaxis.pane.set_linewidth(2)
            ax.zaxis.pane.set_linewidth(2)

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Set axes limits with a small margin
            ax.set_xlim(0, data.shape[1])
            ax.set_ylim(0, data.shape[2])
            ax.set_zlim(0, data.shape[3])


    # Tight layout and show the plot
    plt.tight_layout()

    # Save
    if save_img==True:
        #plt.savefig('Voxel.pdf')
        plt.savefig('Voxel.png', dpi=75)
    plt.show()


filename = './implicit-hair-data/data/nphm/039/39_mesh_orig.obj' # TODO: Change filename here.

mesh = trimesh.load(filename)
angel_voxel = mesh.voxelized(0.005)
print(f'voxel size is {angel_voxel.shape}')
voxel = angel_voxel.matrix

voxel_torch = torch.from_numpy(voxel)[None, None, ...].type(torch.float32).cuda()
gabor = Conv3dGabor(1, 180, size=5, padding='same', device=voxel_torch.device)
F_orient = torch.abs(gabor(voxel_torch))
orientation_maps = F_orient.squeeze().argmax(0)

voxel_orientation = orientation_maps.cpu().unsqueeze(-1).numpy()
plot_3d_voxels(voxel_orientation, sampling_ratio=1.0, skew_factor=1.0)

