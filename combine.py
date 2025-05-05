import numpy as np
from pathlib import Path

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

results_dir = Path("/home/haneesh/courses/6366/project/cracker_box/results")
combined_ply_path = results_dir / "combined_point_cloud.ply"

selected_folders = input("Enter two folder names containing point clouds to merge (comma-separated): ").split(',')
selected_folders = [f.strip() for f in selected_folders if f.strip()]

if len(selected_folders) != 2:
    print("Please provide exactly two folder names.")
    exit()

all_points = []
all_colors = []
processed_folders = []

for folder_name in selected_folders:
    folder = results_dir / folder_name
    if not folder.is_dir():
        print(f"Skipping {folder_name}: Not a valid directory")
        continue

    ply_path = folder / "point_cloud_masked.ply"
    try:

        points, colors = load_ply(str(ply_path))
        all_points.append(points)
        all_colors.append(colors)
        processed_folders.append(folder_name)

    except Exception as e:
        print(f"Error processing {folder_name}: {str(e)}")
        continue

combined_points = np.concatenate(all_points)
combined_colors = np.concatenate(all_colors)

save_ply(str(combined_ply_path), combined_points, combined_colors)
