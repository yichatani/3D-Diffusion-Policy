import os
import sys
import pathlib
import torch
import torch.distributed as dist
from train import TrainDP3Workspace
import hydra
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def setup_distributed(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cfg.global_rank = int(os.environ.get('RANK', 0))
        cfg.world_size = int(os.environ.get('WORLD_SIZE', 1))

        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

        cfg.distributed = True
        print(f"[Distributed] Rank: {cfg.global_rank}, Local rank: {cfg.local_rank}, World size: {cfg.world_size}")
    else:
        cfg.local_rank = 0
        cfg.global_rank = 0
        cfg.world_size = 1
        cfg.distributed = False
        print("[Single GPU] Training with one process on GPU 0")

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

    if cfg.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
