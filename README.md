# sim2real-cs6366
Render an approximate 3d model of a realworld object

# Installation
Follow the instructions in readme of robokit

# Running

### Running segmentation
```
cd robokit
python test_gdino_sam.py
```
make sure to replace the path for the desired data point

### Convertion to pointcloud
```
python depth_to_pc.py
```

### combining two pointclouds
```
python icp_combine.py
```

### Estimating normals and pointcloud to mesh
```
python normals.py
```