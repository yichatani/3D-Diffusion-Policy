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
from process_data import reconstruct_pointcloud,preprocess_point_cloud,preprocess_image

ROOT_DIR = ["/home/ani/3D-Diffusion-Policy/3D-Diffusion-Policy/data",
            "/home/ani/my_Isaac_main/my_Isaac/episodes",
            "/home/ani/astar/my_Isaac/episodes",
            "/home/ani/astar/my_Isaac/episodes/positive",
            "/home/ani/Dataset/episodes/positive",]


def main():

    hdf5_dir = ROOT_DIR[1] + "/positive"
    save_zarr_path = ROOT_DIR[1] + "/positive_cube_ani.zarr"
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
                
                try:
                    pc_raw = reconstruct_pointcloud(rgb[t], depth[t])
                except Exception as e:
                    print(f"[Error] Point cloud reconstruction failed at {t}: {e}")
                    continue

                if pc_raw.shape[0] < 32:
                    print(f"[Warning] Skipping frame {t} in {path} due to empty point cloud.")
                    continue

                img = preprocess_image(rgb[t])
                dep = preprocess_image(np.expand_dims(depth[t], axis=-1)).squeeze(-1)
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
