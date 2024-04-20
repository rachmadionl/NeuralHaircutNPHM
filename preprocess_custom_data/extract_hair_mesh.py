from einops import repeat
import numpy as np
import torch
import trimesh
from torch import nn
import torch.nn.functional as F

from pytorch3d.io import IO, load_obj, load_ply
from pytorch3d.structures import Pointclouds
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_gather, knn_points
 
 
def chamfer_distance_no_reduction(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
):
    """
    Chamfer distance between two pointclouds x and y.
 
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
 
    Returns:
        2-element tuple containing
 
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
 
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
 
    return_normals = x_normals is not None and y_normals is not None
 
    N, P1, D = x.shape
    P2 = y.shape[1]
 
    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]
 
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    # if weights is not None:
    #     if weights.size(0) != N:
    #         raise ValueError("weights must be of shape (N,).")
    #     if not (weights >= 0).all():
    #         raise ValueError("weights cannot be negative.")
    #     if weights.sum() == 0.0:
    #         weights = weights.view(N, 1)
    #         if batch_reduction in ["mean", "sum"]:
    #             return (
    #                 (x.sum((1, 2)) * weights).sum() * 0.0,
    #                 (x.sum((1, 2)) * weights).sum() * 0.0,
    #             )
    #         return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)
 
    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())
 
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)
 
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
 
    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0
 
    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)
 
    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]
 
        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )
 
        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0
 
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)
    
    return (cham_x, cham_y, cham_norm_x, cham_norm_y)
 
 
def pruned_chamfer_loss(x, y, 
                        x_normals=None, y_normals=None,
                        dist_thr=None, normals_thr=None, 
                        mask_x=None, mask_y=None, 
                        device='cuda:0'):
    
    if mask_x is None:
        x_masked = x
        if x_normals:
            x_normals_masked = x_normals.unsqueeze(0)
        else:
            x_normals_masked = None
    else:
        # print('masking')
        x_masked = x[mask_x]
        if x_normals is not None:
            x_normals_masked = x_normals[mask_x].unsqueeze(0)
    
    if mask_y is None:
        y_masked = y
        if y_normals:
            y_normals_masked = y_normals.unsqueeze(0)
        else:
            y_normals_masked = None
    else:
        # print('masking')
        y_masked = y[mask_y]
        if y_normals is not None:
            y_normals_masked = y_normals[mask_y].unsqueeze(0)
    
    cham_x, cham_y, cham_norm_x, cham_norm_y = chamfer_distance_no_reduction(
        x_masked.unsqueeze(0),
        y_masked.unsqueeze(0),
        x_normals=x_normals_masked,
        y_normals=y_normals_masked,
    )
    if x_normals and y_normals:
        cham_x, cham_y, cham_norm_x, cham_norm_y = cham_x[0], cham_y[0], cham_norm_x[0], cham_norm_y[0]
    else:
        cham_x, cham_y = cham_x[0], cham_y[0]

    return cham_x, cham_y, cham_norm_x, cham_norm_y


def main():
    # Load meshes
    neck_color = [164,  39,  79, 255]
    hair_color = [120, 193, 124, 255]
    shirt_color = [249, 209,  44, 255]
    threshold_min = 0.0005

    number = 70
    filename_regs = f'/home/rachmadio/dev/data/NPHM/scan/0{number}/000/registration.ply'
    filename_scan = f'/home/rachmadio/dev/data/NPHM/scan/0{number}/000/scan.obj'
    filename_segm = f'/home/rachmadio/dev/data/NPHM/segmented_meshes/0{number}_exp_000.obj'

    verts_regs, faces_regs = load_ply(filename_regs)
    verts_scan, faces_scan, _ = load_obj(filename_scan)
    mesh_segm = trimesh.load(filename_segm)

    # verts_regs = mesh_regs.vertices.view(np.ndarray)
    # verts_scan = mesh_scan.vertices.view(np.ndarray)
    label_segm = mesh_segm.visual.vertex_colors.view(np.ndarray)

    verts_scan = verts_scan.to('cuda:0')
    verts_regs = verts_regs.to('cuda:0')
    cham_scan, cham_regs, cham_norm_scan, cham_norm_regs = pruned_chamfer_loss(verts_scan, verts_regs)

    verts_scan = verts_scan.cpu()
    cham_scan = cham_scan.cpu()

    filter_neck_idx = (label_segm == neck_color).all(axis=1) == 1
    filter_shirt_idx = (label_segm == shirt_color).all(axis=1) == 1
    filter_out_neck_and_shirt_idx = ~np.logical_or(filter_neck_idx, filter_shirt_idx)

    label_wo_neck_and_shirt = label_segm[filter_out_neck_and_shirt_idx]
    filter_hair_idx = np.where((label_wo_neck_and_shirt == hair_color).all(axis=1) == 1)[0]

    # breakpoint()
    # filter_chamfer_too_far = threshold_max > cham_scan[filter_out_neck_and_shirt_idx]
    filter_chamfer_too_close = cham_scan[filter_out_neck_and_shirt_idx] > threshold_min
    filter_chamfer_idx = filter_chamfer_too_close #  filter_chamfer_too_far.logical_and_(filter_chamfer_too_close)

    verts_scan = verts_scan[filter_out_neck_and_shirt_idx]
    verts_scan_chamfer = verts_scan[filter_chamfer_idx]
    # verts_scan_chamfer = verts_scan_wo_neck_and_shirt[filter_chamfer_idx]
    verts_scan_hair = verts_scan[filter_hair_idx]
    verts_scan_union = torch.unique(torch.cat([verts_scan_chamfer, verts_scan_hair]), dim=0)

    pcl = Pointclouds(points=verts_scan_union.unsqueeze(0))
    IO().save_pointcloud(pcl, f"./results_lower_threshold_remove_max/output_pointcloud_{number}_low_thresh_new.ply")

    # pcl_color = Pointclouds(points=verts_scan_np[filter_color_idx].unsqueeze(0))
    # IO().save_pointcloud(pcl_color, "segmented_pointcloud.ply")

    


if __name__ == '__main__':
    main()