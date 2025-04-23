from datetime import datetime
import dill
import os
import pathlib
from diffusion_policy_3d.dataset.my_dataset import IsaacZarrDataset

ROOT_DIR = str(pathlib.Path(__file__).parent.resolve())
ZARR_PATH = os.path.join(ROOT_DIR, "data", "positive_cube.zarr")

dataset = IsaacZarrDataset(
    zarr_path=ZARR_PATH,
    n_obs_steps=3,
    n_action_steps=6,
    seed=42,
    val_ratio=0.1,
    max_train_episodes=None
)

normalizer = dataset.get_normalizer(mode='limits')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(ROOT_DIR, "data", "outputs", "normalizers", f"normalizer_{timestamp}")
os.makedirs(save_dir, exist_ok=True)


save_path = os.path.join(save_dir, "normalizer.pkl")
with open(save_path, "wb") as f:
    dill.dump(normalizer, f)

print(f"Normalizer saved to {save_path}")
