# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch


from auralflow.models import SeparationModel
from torch.cpu.amp import autocast
from torch.utils.data import DataLoader
from auralflow.visualizer import ProgressBar


def run_validation_step(
    model: SeparationModel, val_dataloader: DataLoader
) -> None:
    """Runs validation loop."""
    max_iters = len(val_dataloader)

    # Set model to validation mode.
    model.eval()
    with ProgressBar(val_dataloader, total=max_iters, desc="valid") as pbar:
        total_loss = mean_loss = 0
        for idx, (mixture, target) in enumerate(pbar):
            with autocast(enabled=model.use_amp):
                with torch.no_grad():
                    # Process data, run forward pass
                    model.set_data(mixture, target)
                    model.test()

                # Compute batch-wise loss.
                batch_loss = model.compute_loss()
                total_loss += batch_loss
                mean_loss = total_loss / (idx + 1)

            # Display loss.
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:6.6f}",
                    "mean_loss": f"{mean_loss:6.6f}",
                }
            )

    # Store epoch-average validation loss.
    model.val_losses.append(mean_loss)
