# train_dp3_main.py

import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from train import TrainDP3Workspace
import hydra
from omegaconf import OmegaConf

@hydra.main(
    version_base=None,
    config_path="diffusion_policy_3d/config",
    config_name="train_dp3"
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
