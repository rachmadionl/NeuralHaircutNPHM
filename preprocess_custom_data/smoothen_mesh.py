from pytorch3d.io import IO
import open3d as o3d


mesh_filename = 'preprocess_custom_data/pipeline_ear_segmented_mesh_sharpened/28_mesh_orig.obj'
mesh = IO().load_mesh(mesh_filename)
L = mesh.laplacian_packed()
n_lapl_iter = 5
mesh_lapl_ver = mesh.verts_packed()
mesh_lapl_face = mesh.faces_packed()
for _ in range(n_lapl_iter):
    mesh_lapl_ver += L.mm(mesh_lapl_ver)    # in-place modification

mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh_lapl_ver.numpy(force=True)),
        triangles=o3d.utility.Vector3iVector(mesh_lapl_face.numpy(force=True))
    )
o3d.io.write_triangle_mesh(f"preprocess_custom_data/pipeline_ear_segmented_mesh_sharpened/28_mesh_smoothed_{n_lapl_iter}_iter.obj", mesh_o3d)