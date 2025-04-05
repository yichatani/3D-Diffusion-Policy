import os
import sys
import pathlib
import torch
import torch.distributed as dist
from train import TrainDP3Workspace
import hydra
from omegaconf import OmegaConf

# 添加当前目录到 sys.path
ROOT_DIR = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(ROOT_DIR)

def setup_distributed(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.global_rank = int(os.environ['RANK'])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        cfg.local_rank = 0
        cfg.global_rank = 0
        cfg.world_size = 1
    print(f"[Rank {cfg.global_rank}] Using GPU {cfg.local_rank}")

@hydra.main(
    version_base=None,
    config_path="diffusion_policy_3d/config",
    config_name="isaac_dp3"
)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    setup_distributed(cfg)
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

# CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 train_Isaac_ddp.py > train_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
if __name__ == "__main__":
    main()
