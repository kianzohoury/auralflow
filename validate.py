# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from models import SeparationModel
from visualizer.progress import ProgressBar
import torch


def cross_validate(model: SeparationModel, val_dataloader: DataLoader) -> None:
    """Validates network updates on the validation set."""
    max_iters = len(val_dataloader)

    model.eval()
    with ProgressBar(val_dataloader, total=max_iters, desc="valid") as pbar:
        total_loss = 0
        for idx, (mixture, target) in enumerate(pbar):
            with autocast(enabled=model.use_amp):

                model.set_data(mixture, target)
                with torch.no_grad():
                    model.test()

                # Compute batch-wise loss.
                batch_loss = model.compute_loss()
                total_loss += batch_loss

            # Display loss.
            pbar.set_postfix({
                "loss": f"{batch_loss:6.6f}",
                "mean_loss": f"{total_loss / (idx + 1):6.6f}"
            })

    # Store epoch-average validation loss.
    model.val_losses.append(total_loss / max_iters)
