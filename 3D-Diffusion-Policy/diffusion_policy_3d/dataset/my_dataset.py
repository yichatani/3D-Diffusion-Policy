import os
import h5py
import numpy as np
import torch
import copy
from typing import Dict
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

class IsaacHDF5Dataset(BaseDataset):
    def __init__(self,
                 hdf5_dir,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.1,
                 max_train_episodes=None,
                 task_name=None):
        super().__init__()
        self.task_name = task_name
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Step 1: 加载所有 episodes
        self.replay_buffer = ReplayBuffer.create_empty()
        episode_files = sorted([os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")])
        for path in episode_files:
            data = self.load_hdf5_episode(path)
            self.replay_buffer.add_episode(data)

        # Step 2: 创建训练/验证 mask
        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask

        # Step 3: 创建序列采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

    def load_hdf5_episode(self, path):
        with h5py.File(path, "r") as f:
            index = f["index"][:]
            agent_pos = f["agent_pos"][:]
            action = f["action"][:]

            # 按需选择相机
            cam = "front"
            rgb = f[f"{cam}/rgb"][:]
            depth = f[f"{cam}/depth"][:]

        # 构造模拟 point cloud 输入，这里直接拼接 rgb + depth 做 placeholder
        # 可以改为 reconstruct_pointcloud(rgb, depth)
        num_frames = len(index)
        rgb = rgb.reshape((num_frames, -1))  # flatten image
        depth = depth.reshape((num_frames, -1))
        point_cloud = np.concatenate([rgb, depth[..., None]], axis=-1)

        return {
            "state": agent_pos.astype(np.float32),       # (T, D_state)
            "action": action.astype(np.float32),         # (T, D_action)
            "point_cloud": point_cloud.astype(np.float32),  # (T, D_pc)
        }

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'point_cloud': self.replay_buffer['point_cloud']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        return {
            'obs': {
                'agent_pos': sample['state'].astype(np.float32),
                'point_cloud': sample['point_cloud'].astype(np.float32),
            },
            'action': sample['action'].astype(np.float32)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
