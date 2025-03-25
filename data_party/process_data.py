import os
import open3d as o3d
import numpy as np
import h5py

ROOT_DIR = "/home/ani/Dataset/episodes"
ROOT_DIR_2 = "/home/ani/astar/my_Isaac/episodes"
POSITIVE_DIR = os.path.join(ROOT_DIR, "positive")
NEGATIVE_DIR = os.path.join(ROOT_DIR, "negative")

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
        # print("First first value:", dset[0])
        return dset[0]
    

def load_episode_data(episode_path):
    """Reads HDF5 data with per-frame, per-camera D_min and D_max."""
    with h5py.File(episode_path, "r") as f:
        index = f["index"][:]
        agent_pos = f["agent_pos"][:]
        action = f["action"][:]

        cameras = ["front", "in_hand", "up"]
        num_frames = len(index)

        cameras_data = [{} for _ in range(num_frames)]

        for i in range(num_frames):
            frame_data = {}
            for cam in cameras:
                rgb = f[f"{cam}/rgb"][i]
                depth_uint16 = f[f"{cam}/depth"][i]

                # Read per-frame D_min, D_max
                D_min = f[f"{cam}/D_min"][i]
                D_max = f[f"{cam}/D_max"][i]

                # Decode to float depth
                # depth = (depth_uint16.astype(np.float32) / 65535.0) * (D_max - D_min) + D_min
                depth = depth_uint16

                frame_data[cam] = {
                    "rgb": rgb,
                    "depth": depth,
                    "D_min": D_min,
                    "D_max": D_max
                }

            cameras_data[i] = frame_data

    return index, agent_pos, action, cameras_data



def reconstruct_pointcloud(rgb, depth):
    """
    Reconstruct point cloud from colors and depths.
    """
    colors = rgb / 255.0
    depths = depth
    camera_matrix = [[531.29, 0.0, 224], [0.0, 531.29, 224], [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    scale = 1000.0

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    print("points_z:", points_z)
    points_x = (xmap - cx) / fx * points_z
    print("points_x:", points_x)
    points_y = (ymap - cy) / fy * points_z
    print("points_y:", points_y)

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 2)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # print("points shape:", points.shape[0])
    points = points[mask].astype(np.float32)
    # print("points shape:", points.shape[0])
    colors = colors[mask].astype(np.float32)
    colors = colors[:, :3]  # remove transparent Alpha channel

    # print("points shape:", points.shape, "colors shape:", colors.shape)
    if points.shape[0] == 0:
        print("Warning: Empty point cloud!")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([cloud])

    return cloud




if __name__ == "__main__":
    episode_path = ROOT_DIR_2 + "/episode_0.h5"

    index, agent_pos, action, cameras_data = load_episode_data(episode_path)
    print("index:", index.shape)
    print("agent_pos:", agent_pos.shape)
    print("action:", action.shape)
    print("cameras_data:", cameras_data[2]["front"]["rgb"].shape)
    print("cameras_data:", cameras_data[2]["front"]["depth"].shape)

    # read_structure(path)
    # depth = read_values(path,"up/depth")
    # rgb = read_values(path,"up/rgb")
    # print(rgb.shape, depth.shape)

    # point_cloud = reconstruct_pointcloud(cameras_data[60]["front"]["rgb"], cameras_data[60]["front"]["depth"])
    # point_cloud = reconstruct_pointcloud(cameras_data[20]["in_hand"]["rgb"], cameras_data[20]["in_hand"]["depth"])

    # colors = read_values(path, "color")
    # depths = read_values(path, "depth")
    # reconstruct_pointcloud(colors, depths)
