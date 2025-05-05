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

def save_ply(filename, points, colors):
    num_points = len(points)
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def create_open3d_pointcloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  

def icp_registration(source_pcd, target_pcd, threshold=0.02, max_iterations=2000):
    init_trans = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return reg_p2p.transformation

results_dir = Path("/home/haneesh/courses/6366/project/cracker_box/results")
combined_ply_path = results_dir / "combined_point_cloud.ply"

selected_folders = input("Enter folder names containing point clouds to merge (comma-separated): ").split(',')
selected_folders = [f.strip() for f in selected_folders if f.strip()]


all_pointclouds = []
processed_folders = []

for folder_name in selected_folders:
    folder = results_dir / folder_name

    ply_path = folder / "point_cloud_masked.ply"
    try:
        if not ply_path.exists():
            print(f"Skipping {folder_name}: {ply_path} does not exist")
            continue

        points, colors = load_ply(str(ply_path))
        pcd = create_open3d_pointcloud(points, colors)
        all_pointclouds.append((pcd, points, colors))
        processed_folders.append(folder_name)

    except Exception as e:
        print(f"Error processing {folder_name}: {str(e)}")
        continue


combined_points = all_pointclouds[0][1]  # Original points
combined_colors = all_pointclouds[0][2]  # Original colors
combined_pcd = all_pointclouds[0][0]     

for i, (source_pcd, source_points, source_colors) in enumerate(all_pointclouds[1:], 1):
    print(f"Processing point cloud {i+1}/{len(all_pointclouds)} from folder: {processed_folders[i]}")

    source_pcd_down = source_pcd.voxel_down_sample(voxel_size=0.005)
    combined_pcd_down = combined_pcd.voxel_down_sample(voxel_size=0.005)
    transformation = icp_registration(source_pcd_down, combined_pcd_down, threshold=0.02)

    source_points_homogeneous = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    transformed_source_points = (transformation @ source_points_homogeneous.T).T[:, :3]
    combined_points = np.concatenate((combined_points, transformed_source_points))
    combined_colors = np.concatenate((combined_colors, source_colors))

    combined_pcd = create_open3d_pointcloud(combined_points, combined_colors)

save_ply(str(combined_ply_path), combined_points, combined_colors)
