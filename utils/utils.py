import torch

from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Optional, Union


def checkpoint_name(name, global_steps, loss_fn, loss):
    return f"{name}_{global_steps}_{loss_fn}={loss}.pth"


def checkpoint_handler(model_state: dict, checkpoint_dir: Optional[str] = None,
                       save_best: bool = True, display: bool = False):

    checkpoint_path = Path(checkpoint_dir)

    # Create model-specific checkpoint folder if one does not already exist in
    # the project path -- useful for training different base models.
    model_base = model_state['model_base']
    model_path = checkpoint_path.joinpath(model_base)
    model_path.mkdir(parents=True, exist_ok=True)

    model_path_latest = model_path.joinpath('latest.pth')
    model_path_best = model_path.joinpath('best.pth')

    # Save latest version.
    torch.save(model_state, model_path_best.absolute())

    if save_best:
        if not model_path_best.exists():
            torch.save(model_state, model_path_best.absolute())
        else:
            best_val = torch.load(model_path_best.absolute())['best_val_loss']
            if model_state['final_val_loss'] < best_val:
                torch.save(model_state, model_path_best.absolute())
    if display:
        print(f"Checkpoint saved successfully.")