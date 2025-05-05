import cv2
import numpy as np
from pathlib import Path
from convertion import t265_to_d435
import transforms3d

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) 
    return xyz_img

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

results_dir = Path("/home/haneesh/courses/6366/project/cracker_box/results")
fx, fy = 606.64532471, 606.66217041
cx, cy = 320.29736328, 246.94914246

selected_folders = input("Enter folder names to process (comma-separated): ").split(',')
selected_folders = [f.strip() for f in selected_folders if f.strip()] if selected_folders[0] else []

processed_folders = []
for folder in results_dir.iterdir():
    if not folder.is_dir() or (selected_folders and folder.name not in selected_folders):
        continue

    depth_image_path = folder / "depth.png"
    rgb_image_path = folder / "rgb.png"
    mask_image_path = folder / "mask.png"
    ply_output_path = folder / "point_cloud_masked.ply"
    pose_path = folder / "pose.npz"

    try:
        depth_img = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(str(rgb_image_path))
        mask_img = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)

        if depth_img is None or rgb_img is None or mask_img is None:
            print(f"Skipping {folder.name}: Could not load depth, RGB, or mask image")
            continue

        if depth_img.shape[:2] != rgb_img.shape[:2] or depth_img.shape[:2] != mask_img.shape[:2]:
            print(f"Skipping {folder.name}: Depth, RGB, and mask images must have the same resolution")
            continue

        height, width = depth_img.shape
        depth_img = depth_img / 1000.0
        xyz_img = compute_xyz(depth_img, fx, fy, cx, cy, height, width)

        xyz_flat = xyz_img.reshape(-1, 3)
        mask_flat = mask_img.flatten()
        colors = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).reshape(-1, 3)

        valid = (xyz_flat[:, 2] > 0) & (mask_flat > 0) & (xyz_flat[:, 2] < 1)
        points = xyz_flat[valid]
        p_copy = np.copy(points)
        pose_data = np.load(pose_path)
        quaternion = pose_data["orientation"]
        position =  pose_data["position"]

        quat_wxyz = np.roll(quaternion, 1)  # [x, y, z, w] -> [w, x, y, z]
        T = transforms3d.affines.compose(position, transforms3d.quaternions.quat2mat(quat_wxyz), np.ones(3))
        for i, point in enumerate(points):
            point_hom = np.append(point, 1)  
            transformed = T @ t265_to_d435 @ point_hom  
            p_copy[i] = transformed[:3]  
        colors_valid = colors[valid]



        save_ply(str(ply_output_path), p_copy, colors_valid)
        processed_folders.append(folder.name)

    except Exception as e:
        print(f"Error processing {folder.name}: {str(e)}")
        continue
