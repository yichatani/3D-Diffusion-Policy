# save as: preprocess_hdf5_to_zarr.py

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
from process_data import reconstruct_pointcloud

ROOT_DIR = ["/home/ani/3D-Diffusion-Policy/3D-Diffusion-Policy/data/episodes/positive",
            "/home/ani/my_Isaac_main/my_Isaac/episodes",
            "/home/ani/astar/my_Isaac/episodes",
            "/home/ani/astar/my_Isaac/episodes/positive",
            "/home/ani/Dataset/episodes/positive"]

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
    WORK_SPACE = [[-0.12, 1.12], [-0.30, 0.50], [0.128, 1.5]]
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

def main():

    hdf5_dir = ROOT_DIR[2] + "/positive"
    save_zarr_path = ROOT_DIR[2] + "/positive_Z.zarr"
    camera = 'front'  # change to 'in_hand' or 'up' if needed

    episode_paths = sorted([
        os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir)
        if f.endswith('.h5') or f.endswith('.hdf5')
    ])
    
    if os.path.exists(save_zarr_path):
        cprint(f"Warning: Zarr directory already exists: {save_zarr_path}", "red")
        cprint("Overwriting it...", "red")
        os.system(f"rm -rf {save_zarr_path}")

    cprint(f"Saving to Zarr: {save_zarr_path}", "green")
    zarr_root = zarr.group(save_zarr_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # Buffers
    img_list, depth_list, pc_list = [], [], []
    state_list, action_list = [], []
    episode_ends = []

    total_count = 0

    for path in tqdm.tqdm(episode_paths, desc="Processing Episodes"):
        with h5py.File(path, "r") as f:
            agent_pos = f["agent_pos"][:]
            action = f["action"][:]
            rgb = f[f"{camera}/rgb"][:] # range in [0,255]
            depth = f[f"{camera}/depth"][:]
            

            T = agent_pos.shape[0]
            for t in range(T):
                img = preprocess_image(rgb[t])
                dep = preprocess_image(np.expand_dims(depth[t], axis=-1)).squeeze(-1)
                
                pc_raw = reconstruct_pointcloud(rgb[t], depth[t])
                if pc_raw.shape[0] < 32:
                    print(f"[Warning] Skipping frame {t} in {path} due to empty point cloud.")
                    continue

                pc = preprocess_point_cloud(pc_raw, use_cuda=True)
              

                img_list.append(img)
                depth_list.append(dep)
                pc_list.append(pc)
                state_list.append(agent_pos[t])
                action_list.append(action[t])
                total_count += 1


        episode_ends.append(total_count)

    # Stack all
    img_arr = np.stack(img_list, axis=0).astype('uint8')
    dep_arr = np.stack(depth_list, axis=0).astype('float32')
    pc_arr = np.stack(pc_list, axis=0).astype('float32')
    state_arr = np.stack(state_list, axis=0).astype('float32')
    action_arr = np.stack(action_list, axis=0).astype('float32')
    episode_ends_arr = np.array(episode_ends, dtype='int64')

    # Save zarr
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zarr_data.create_dataset('img', data=img_arr, compressor=compressor, chunks=(100, *img_arr.shape[1:]))
    zarr_data.create_dataset('depth', data=dep_arr, compressor=compressor, chunks=(100, *dep_arr.shape[1:]))
    zarr_data.create_dataset('point_cloud', data=pc_arr, compressor=compressor, chunks=(100, *pc_arr.shape[1:]))
    zarr_data.create_dataset('state', data=state_arr, compressor=compressor, chunks=(100, state_arr.shape[1]))
    zarr_data.create_dataset('action', data=action_arr, compressor=compressor, chunks=(100, action_arr.shape[1]))
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arr, dtype='int64')

    cprint(f"[Done] Saved Zarr with {total_count} frames, {len(episode_ends)} episodes", "green")

if __name__ == '__main__':
    main()
