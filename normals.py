import numpy as np
from pathlib import Path
import open3d as o3d

def load_ply(filename):
    points = []
    colors = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        header_end = lines.index("end_header\n") + 1
        for line in lines[header_end:]:
            parts = line.strip().split()
            x, y, z = map(float, parts[:3])
            r, g, b = map(int, parts[3:6])
            points.append([x, y, z])
            colors.append([r, g, b])
    return np.array(points), np.array(colors)

def filter_outliers(pcd, nb_neighbors=200, std_ratio=2.0):
    print(f"Before outlier removal: {len(pcd.points)} points")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"After outlier removal: {len(pcd.points)} points")
    return pcd

results_dir = Path("/home/haneesh/courses/6366/project/cracker_box/results/00043/")
combined_ply_path = results_dir / "point_cloud_masked.ply"
mesh_ply_path = results_dir / "mesh.ply"

try:
    # Load point cloud
    points, colors = load_ply(str(combined_ply_path))
    print(f"Loaded point cloud with {len(points)} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize RGB to [0, 1]

    pcd = filter_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    if len(pcd.points) < 100:
        raise ValueError("Point cloud has too few points after outlier removal. Adjust nb_neighbors or std_ratio.")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=2000))

    if not pcd.has_normals():
        raise ValueError("Normal estimation failed. Check point cloud density or adjust radius/max_nn.")

    o3d.visualization.draw_geometries([pcd], point_show_normal=True, window_name="Point Cloud with Normals")

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=2)
    mesh_vertex_colors = np.zeros((len(mesh.vertices), 3))  # Initialize vertex colors
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i, vertex in enumerate(np.asarray(mesh.vertices)):
        [k, idx, _] = kdtree.search_knn_vector_3d(vertex, 1)
        mesh_vertex_colors[i] = np.asarray(pcd.colors)[idx[0]]  # Assign color from nearest point
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)

    o3d.visualization.draw_geometries([mesh], window_name="Colored Mesh", width=800, height=600)


except Exception as e:
    print(f"Error processing point cloud: {str(e)}")