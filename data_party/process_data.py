import os
import open3d as o3d
import numpy as np
import h5py
import os
import h5py
import zarr
import numpy as np
import torch
import torchvision
import tqdm
import pickle
import open3d as o3d
from termcolor import cprint
from pathlib import Path
from pytorch3d.ops import sample_farthest_points
from visualizer.pointcloud import visualize_pointcloud

ROOT_DIR = ["/home/ani/3D-Diffusion-Policy/3D-Diffusion-Policy/data/episodes/positive",
            "/home/ani/my_Isaac_main/my_Isaac/episodes",
            "/home/ani/astar/my_Isaac/episodes",
            "/home/ani/astar/my_Isaac/episodes/positive",
            "/home/ani/Dataset/episodes/positive",
            "/home/ani/Dataset/positive_1.zarr"]
# ROOT_DIR_2 = "/home/ani/astar/my_Isaac/episodes"
# ROOT_DIR_3 = "/home/ani/Dataset/episodes/positive"
POSITIVE_DIR = os.path.join(ROOT_DIR[1], "positive")
NEGATIVE_DIR = os.path.join(ROOT_DIR[1], "negative")

def read_structure(path):
    """
    Read the data structure of h5 files.
    """
    with h5py.File(path, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)
        f.visititems(print_hdf5_structure)


def read_values(path, key):
    """
    Read the key values. 
    """
    with h5py.File(path, 'r') as f:
        dset = f[key]
        print("Dataset shape:", dset.shape)
        print("First first value:", dset[0])
        return dset[0]
    




def reconstruct_pointcloud(rgb, depth, visualize=False):
    """
    Reconstruct point cloud from RGB + depth.

    Returns:
        point_cloud: (Np, 6) numpy array, columns: [x, y, z, r, g, b]
    """
    # Normalize RGB to [0,1]
    colors = rgb[..., :3] / 255.0
    depths = depth
    camera_matrix = [[531.29, 0.0, 224], [0.0, 531.29, 224], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    scale = 1.0  # if your depth is in mm, scale it to meters

    # Construct pixel grid
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # mask = (points_z > 0) & (points_z < 2)  # optional: crop invalid range
    mask = (points_z > 0) & (points_z < 3)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    if points.shape[0] == 0:
        print("Warning: Empty point cloud!")
        return np.zeros((0, 6), dtype=np.float32)

    if visualize:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([cloud])

    # Combine [x, y, z, r, g, b]
    point_cloud = np.concatenate([points, colors], axis=1)  # shape: (Np, 6)
    return point_cloud


def preprocess_image(image, img_size=84):
    image = image.astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    return image

def preprocess_point_cloud(points, num_points=1024, use_cuda=True):
    extrinsics_matrix = np.array([
        [-0.61193014,  0.2056703,  -0.76370232,  2.22381139],
        [ 0.78640693,  0.05530829, -0.61522771,  1.06986129],
        [-0.084295,   -0.97705717, -0.19558536,  0.90482569],
        [ 0.,          0.,          0.,          1.        ],
    ])

    WORK_SPACE = [[-0.12, 1.12], [-0.40, 0.80], [0.128, 1.5]]
    # WORK_SPACE = [[-0.12, 1.12], [-0.30, 0.80], [0, 1.5]]
    # WORK_SPACE = [[-2, 2], [-2, 2], [-2, 2]]

    # point_xyz = points[..., :3] * 0.00025
    point_xyz = points[..., :3]
    point_hom = np.concatenate([point_xyz, np.ones((point_xyz.shape[0], 1))], axis=1)
    point_xyz = point_hom @ extrinsics_matrix.T
    points[..., :3] = point_xyz[..., :3]

    mask = (
        (points[:, 0] > WORK_SPACE[0][0]) & (points[:, 0] < WORK_SPACE[0][1]) &
        (points[:, 1] > WORK_SPACE[1][0]) & (points[:, 1] < WORK_SPACE[1][1]) &
        (points[:, 2] > WORK_SPACE[2][0]) & (points[:, 2] < WORK_SPACE[2][1])
    )
    points = points[mask]
    if points.shape[0] == 0:
        raise ValueError("All points filtered out by WORK_SPACE constraints.")

    if use_cuda:
        pts_tensor = torch.from_numpy(points[:, :3]).unsqueeze(0).cuda()
    else:
        pts_tensor = torch.from_numpy(points[:, :3]).unsqueeze(0)
    sampled_pts, indices = sample_farthest_points(pts_tensor, K=num_points)
    sampled_pts = sampled_pts.squeeze(0).cpu().numpy()
    indices = indices.cpu().squeeze(0)
    rgb = points[indices.numpy(), 3:]
    return np.hstack((sampled_pts, rgb))


def visualize_h5_frame(camera,t,episode_path):

    # for path in tqdm.tqdm(episode_paths, desc="Processing Episodes"):
    with h5py.File(episode_path, "r") as f:
        rgb = f[f"{camera}/rgb"][:]
        depth = f[f"{camera}/depth"][:]
        
        print(f"{camera}/rgb", rgb.shape)
        pc_raw = reconstruct_pointcloud(rgb[t], depth[t])
        if pc_raw.shape[0] < 32:
            raise ValueError(f"[Warning] Skipping frame {t} in {episode_path} due to empty point cloud.")
            
        pc = preprocess_point_cloud(pc_raw, use_cuda=True)
        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(pc[:, :3])
        # cloud.colors = o3d.utility.Vector3dVector(pc[:, 3:])
        # o3d.visualization.draw_geometries([cloud])
        visualize_pointcloud(pc)


def visualize_zarr_frame(zarr_path, frame_idx):
    """
    从 Zarr 文件中读取并可视化某一帧的点云。
    """
    cprint(f"Loading Zarr dataset from: {zarr_path}", "cyan")
    root = zarr.open(zarr_path, mode='r')
    
    try:
        pc_arr = root['data']['point_cloud']
    except Exception as e:
        cprint(f"Error loading point cloud dataset: {e}", "red")
        return

    if frame_idx >= len(pc_arr):
        cprint(f"Invalid frame_idx: {frame_idx}, total frames: {len(pc_arr)}", "red")
        return

    pc = pc_arr[frame_idx]  # shape: (N, 6), columns: [x, y, z, r, g, b]
    visualize_pointcloud(pc)

def read_zarr_meta(zarr_path, meta_key:str) -> None:
    cprint(f"Loading Zarr dataset from: {zarr_path}", "cyan")
    root = zarr.open(zarr_path, mode='r')
    try:
        meta_data = root['meta'][meta_key]
    except Exception as e:
        cprint(f"Error loading meta data: {e}", "red")
        return
    print(meta_data[:])

if __name__ == "__main__":
    episode_path = ROOT_DIR[2] + "/episode_0.h5"
    zarr_path = ROOT_DIR[4]
    print(episode_path)
    # exit()
    visualize_h5_frame("front",139,episode_path)

    # read_zarr_meta(zarr_path,"episode_ends")
    # visualize_zarr_frame(zarr_path,747)
    

    # read_values(episode_path,"label")
    # read_values(episode_path,"action")
    # read_structure(episode_path)

  