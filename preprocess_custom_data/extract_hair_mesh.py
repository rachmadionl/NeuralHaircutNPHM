import argparse
import copy
import os
import pickle as pk

from einops import repeat
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import trimesh
from torch import nn
import torch.nn.functional as F

from pytorch3d.io import IO, load_obj, load_ply, save_obj
from pytorch3d.structures import Meshes, Pointclouds
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


def dilate_vertex_mask(edges, mask):
    # edges: (M, 2) torch tensor long
    # mask: (N,) torch tensor bool (1 = keep; 0 = omit)

    mask_dilated = mask.clone().bool()
    edges_crossing = mask_dilated[edges[:, 0]] != mask_dilated[edges[:, 1]]
    mask_dilated[edges[edges_crossing, 0]] = True
    mask_dilated[edges[edges_crossing, 1]] = True
    return mask_dilated


def inv(binary_mask: np.ndarray) -> np.ndarray:
    return ~binary_mask


def get_border_edges_and_extract_the_vertice(mesh):
    single_triangle_edges = []
    edge_triangle_count = {}

    for face in np.asarray(mesh.triangles):
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)]) if face[i] < face[(i + 1) % len(face)] else (face[(i + 1) % len(face)], face[i])
            edge_triangle_count[edge] = edge_triangle_count.get(edge, 0) + 1

    for edge, count in edge_triangle_count.items():
        if count == 1:
            single_triangle_edges.append(edge)
    
    single_triangle_edges = np.array(single_triangle_edges)
    verts_idx = np.unique(np.concatenate([single_triangle_edges[:, i] for i in range(len(single_triangle_edges[0]))]))
    # breakpoint()
    return verts_idx


def remove_vertices_and_corresponding_faces(ver, faces, mask, ver_n = None):
    # ver: (N, 3) torch tensor float32
    # faces: (N, 3) torch tensor long
    # mask: (N,) torch tensor bool (1 = keep; 0 = omit)

    if ver_n is not None:
        assert len(ver) == len(ver_n)
    ver_to_keep = torch.arange(ver.shape[0]).to(ver.device)[mask]
    ver_new = ver[ver_to_keep]
    new2old = torch.arange(ver.shape[0]).to(ver.device)[mask]
    old2new = torch.full((ver.shape[0],), -1, dtype=torch.long).to(ver.device)
    old2new[new2old] = torch.arange(new2old.shape[0]).to(ver.device)
    faces_new = faces[torch.all(mask[faces], dim=1)]
    faces_new = old2new[faces_new]
    if ver_n is not None:
        ver_n_new = ver_n[ver_to_keep]
        return ver_new, ver_n_new, faces_new
    return ver_new, faces_new

# List of Segmentation Mesh Color:
# array([
#    [ 22,  90, 101, 255], -> Right Ear
#    [ 22, 143, 184, 255], -> Lips 
#    [ 58,  97,   6, 255],
#    [ 73,  62, 222, 255], -> Nose
#    [ 79, 105,  62, 255], -> Right Eyebrow
#    [100,   0,   1, 255], -> Left Eye
#    [104, 199,   8, 255], -> Lower lip
#    [109,  26, 150, 255], -> Left Eyebrow
#    [109, 149, 195, 255], -> Upper lip
#    [120, 193, 124, 255], -> Hair
#    [135,  86,  78, 255],
#    [138,  31,  70, 255],
#    [164,  39,  79, 255], -> Neck
#    [177,   5, 249, 255], -> Enclosure
#    [181, 127,  49, 255], -> Left Ear
#    [201, 212, 140, 255], -> Left Eye
#    [211,  67,  14, 255], -> Face
#    [249, 209,  44, 255]  -> Shirt
#    ], dtype=uint8)


def main(args, number):
    # Load meshes
    print(f'Extracting hair on subject number {number} ...')
    neck_color = [164,  39,  79, 255]
    hair_color = [120, 193, 124, 255]
    shirt_color = [249, 209,  44, 255]
    face_color = [211, 67, 14, 255]
    ear_left_color = [181, 127,  49, 255]
    ear_right_color = [22,  90, 101, 255]

    threshold_min = 0.0005
    threshold_ear_min = 0.0005

    filename_regs = f'/home/rachmadio/dev/data/NPHM/scan/0{number}/000/registration.ply'
    filename_scan = f'/home/rachmadio/dev/data/NPHM/scan/0{number}/000/scan.obj'
    filename_segm = f'/home/rachmadio/dev/data/NPHM/segmented_meshes/0{number}_exp_000.obj'
    filename_flame = f'/home/rachmadio/dev/data/NPHM/scan/0{number}/000/flame.ply'

    verts_regs, faces_regs = load_ply(filename_regs)
    verts_scan, faces_scan, aux_scan = load_obj(filename_scan)
    verts_flame, faces_flame = load_ply(filename_flame)
    aux_scan_normals = aux_scan.normals
    mesh_segm = trimesh.load(filename_segm)

    mesh_scan = Meshes(verts=[verts_scan], faces=[faces_scan.verts_idx])
    # verts_regs = mesh_regs.vertices.view(np.ndarray)
    # verts_scan = mesh_scan.vertices.view(np.ndarray)
    label_segm = mesh_segm.visual.vertex_colors.view(np.ndarray)

    verts_scan = verts_scan.to('cuda:0')
    verts_regs = verts_regs.to('cuda:0')
    verts_flame =verts_flame.to('cuda:0')
    cham_scan, _, _, _ = pruned_chamfer_loss(verts_scan, verts_regs)

    verts_scan = verts_scan.cpu()
    cham_scan = cham_scan.cpu()

    neck_idx = (label_segm == neck_color).all(axis=1) == 1
    shirt_idx = (label_segm == shirt_color).all(axis=1) == 1
    face_idx = (label_segm == face_color).all(axis=1) == 1
    no_face_neck_shirt_idx = ~(np.logical_or(np.logical_or(neck_idx, shirt_idx), face_idx))

    label_no_neck_and_shirt = label_segm[no_face_neck_shirt_idx]
    filter_hair_idx = np.where((label_no_neck_and_shirt == hair_color).all(axis=1) == 1)[0]

    # Ears with segmented Mesh
    if not args.use_flame:
        ear_left_idx = (label_segm == ear_left_color).all(axis=1) == 1
        ear_right_idx = (label_segm == ear_right_color).all(axis=1) == 1
        ears_idx = np.logical_or(ear_left_idx, ear_right_idx)

        ears_shrunked_idx =  inv(dilate_vertex_mask(mesh_scan.edges_packed(), torch.from_numpy(inv(ears_idx))).numpy())
        ears_shrunked_filtered_idx = ears_shrunked_idx[no_face_neck_shirt_idx]

    else:
        # Ears with FLAME Region
        with open('flame_masks.pkl', 'rb') as fin:
            flame_region_masks = pk.load(fin)
        
        ear_left_flame_idx = flame_region_masks['left_ear']
        ear_right_flame_idx = flame_region_masks['right_ear']
        ears_flame_idx = np.concatenate([ear_left_flame_idx, ear_right_flame_idx])
    # ears_flame_mask = torch.zeros(verts_flame.size(0)).to(verts_flame.device)
    # ears_flame_mask[ears_flame_idx] = 1


    # filter_chamfer_too_far = threshold_max > cham_scan[filter_out_neck_and_shirt_idx]
    filter_chamfer_too_close = cham_scan[no_face_neck_shirt_idx] > threshold_min
    filter_chamfer_idx = filter_chamfer_too_close #  filter_chamfer_too_far.logical_and_(filter_chamfer_too_close)


    verts_scan = verts_scan[no_face_neck_shirt_idx]
    aux_scan_normals = aux_scan_normals[no_face_neck_shirt_idx]

    verts_scan_chamfer = verts_scan[filter_chamfer_idx]
    aux_scan_normals_chamfer = aux_scan_normals[filter_chamfer_idx]
    # verts_scan_chamfer = verts_scan_wo_neck_and_shirt[filter_chamfer_idx]
    verts_scan_hair = verts_scan[filter_hair_idx]

    filter_chamfer_idx_value = filter_chamfer_idx.numpy().nonzero()[0]
    filter_hair_idx_value = filter_hair_idx.nonzero()[0]
    aux_scan_normals_hair = aux_scan_normals[filter_hair_idx]

    verts_scan_cat = torch.cat([verts_scan_chamfer, verts_scan_hair])
    aux_scan_normals_cat = np.concatenate([aux_scan_normals_chamfer, aux_scan_normals_hair])
    # verts_scan_union = torch.unique(torch.cat([verts_scan_chamfer, verts_scan_hair]), dim=0)
    # verts_scan_union_idx = np.unique(np.concatenate([filter_chamfer_idx_value, filter_hair_idx_value]), axis=0)
    verts_scan_union_np, idx = np.unique(verts_scan_cat.numpy(), axis=0, return_index=True)
    
    verts_scan_union = verts_scan_cat[idx]
    aux_normals_union = aux_scan_normals_cat[idx]

    
    if not args.use_flame:  # Chamfer Ears with Segmented Mesh
        print('Use Mesh Ear Segmentaton')
        cham_scan_union, _, _, _ = pruned_chamfer_loss(verts_scan_union.to('cuda'), verts_scan[ears_shrunked_filtered_idx].to('cuda'))
        cham_scan_union = cham_scan_union.to('cpu')
        verts_scan_union = verts_scan_union.to('cpu')

    else:  # Chamfer Ears with FLAME
        print('Use FLAME for Ear Segmentaton')
        cham_scan_union, _, _, _ = pruned_chamfer_loss(verts_scan_union.to('cuda'), verts_flame[ears_flame_idx])
        cham_scan_union = cham_scan_union.to('cpu')
        verts_scan_union = verts_scan_union.to('cpu')

    filter_ear_chamfer = cham_scan_union > threshold_ear_min
    verts_scan_final = verts_scan_union[filter_ear_chamfer]
    aux_normals_final = aux_normals_union[filter_ear_chamfer]

    filter_y = verts_scan_final[:, 1] > (verts_scan_final[:, 1].median() - 0.55)
    verts_scan_final = verts_scan_final[filter_y]
    aux_normals_final = aux_normals_final[filter_y]
    
    pcl = Pointclouds(points=verts_scan_final.unsqueeze(0))
    save_filename = f"./{args.out_folder}/{number}_pcd.ply"

    IO().save_pointcloud(pcl, save_filename)
    print(f'Done! Saving the results to {save_filename}\n')

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(verts_scan_final)
    pcd.normals = o3d.utility.Vector3dVector(aux_normals_final)
    # breakpoint()
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1e-6, max_nn=30))
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(100)
    # o3d.visualization.draw_geometries([pcd])

    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.07)
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.5)
    # o3d.visualization.draw_geometries([cl])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    # mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.639, 0.639, 0.592])
    # print('Displaying reconstructed mesh ...')
    # o3d.visualization.draw([mesh])

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    # o3d.visualization.draw_geometries([density_mesh])

    print('remove low density vertices')
    verts = np.asarray(mesh.vertices)
    vertices_low_density_mask = densities < np.quantile(densities, 0.01)
    vertices_low_density = verts[vertices_low_density_mask]

    if np.abs(np.max(vertices_low_density[:, 1]) - np.max(verts[:, 1])) <= 0.2: # Low Density Area at Top, do not filter them out.
        vertices_low_y_mask = verts[:, 1] < (np.max(vertices_low_density[:, 1]) - 0.1)
        vertices_to_remove = np.multiply(vertices_low_density_mask, vertices_low_y_mask)
    else:
        vertices_to_remove = vertices_low_density_mask

    pcd_low_density = o3d.geometry.PointCloud()
    pcd_low_density.points = o3d.utility.Vector3dVector(verts[vertices_to_remove])
    pcd_low_density.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw([mesh, pcd_low_density])
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # print("Show mesh with small clusters removed")
    # triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
    # mesh.remove_triangles_by_mask(triangles_to_remove)

    print("Show largest cluster")
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    # o3d.visualization.draw_geometries([mesh])

    verts_border_idx = get_border_edges_and_extract_the_vertice(mesh)
    pcd_border = o3d.geometry.PointCloud()
    pcd_border.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts_border_idx])
    pcd_border.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw([mesh, pcd_border])

    verts = torch.from_numpy(np.asarray(mesh.vertices))
    faces = torch.from_numpy(np.asarray(mesh.triangles))
    verts_n = torch.from_numpy(np.asarray(mesh.vertex_normals))
    mesh_torch = Meshes(verts=[verts], faces=[faces])
    verts_border_mask = torch.zeros(verts.size(0))
    verts_border_mask[verts_border_idx] = 1
    for _ in range(3):
        verts_border_mask = dilate_vertex_mask(mesh_torch.edges_packed(), verts_border_mask)

    verts_final, verts_n_final, faces_final = remove_vertices_and_corresponding_faces(verts, faces.long(), ~verts_border_mask, verts_n)
    mesh_filename = f"./{args.out_folder}/{number}_mesh.obj"
    verts_n_final = F.normalize(verts_n_final, p=2, dim=1)
    # mesh_final = Meshes(verts=[verts_final], verts_normals=[verts_n_final], faces=[faces_final])
    # IO().save_mesh(mesh_final, mesh_filename)
    mesh_reconstructed = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_final.numpy(force=True)),
        triangles=o3d.utility.Vector3iVector(faces_final.numpy(force=True))
    )
    # mesh_reconstructed.vertex_normals = o3d.utility.Vector3dVector(verts_n_final.numpy(force=True))
    o3d.io.write_triangle_mesh(mesh_filename, mesh_reconstructed)

    verts_scan, faces_scan, aux_scan = load_obj(filename_scan, device='cuda:0')
    chamf_scan, _, _, _ = pruned_chamfer_loss(verts_scan.to(torch.float32), verts_final.to(device='cuda:0', dtype=torch.float32))
    chamf_mask = chamf_scan <= 1e-4
    verts_orig, faces_orig = remove_vertices_and_corresponding_faces(verts_scan, faces_scan.verts_idx.long(), chamf_mask)
    # mesh_orig = Meshes(verts=[verts_orig], faces=[faces_orig])
    # mesh_orig
    # IO().save_mesh(mesh_orig, f"./{args.out_folder}/{number}_mesh_orig.obj")
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_orig.numpy(force=True)),
        triangles=o3d.utility.Vector3iVector(faces_orig.numpy(force=True))
    )
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_o3d.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_o3d.remove_triangles_by_mask(triangles_to_remove)
    o3d.io.write_triangle_mesh(f"./{args.out_folder}/{number}_mesh_orig.obj", mesh_o3d)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nphm_folder', type=str, default='/home/rachmadio/dev/data/NPHM/scan')
    parser.add_argument('--out_folder', type=str, default='pipeline_orig_ear_segmented_mesh')
    parser.add_argument('--use_flame', type=bool, default=False)
    parser.add_argument('--number', type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)

    if not args.number:
        nphm_folder = args.nphm_folder

        folders = sorted(os.listdir(nphm_folder))
        print(f'There are {len(folders)} subjects.\n')
        for folder in folders:
            main(args, int(folder))
    else:
        main(args, args.number)