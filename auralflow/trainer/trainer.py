# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from .callbacks import TrainingCallback
from auralflow.models import SeparationModel
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from auralflow.visualizer import ProgressBar


def run_training_step(
    model: SeparationModel,
    start_epoch: int,
    stop_epoch: int,
    global_step: int,
    train_dataloader: DataLoader,
    callback: TrainingCallback,
) -> None:
    """Runs training loop."""
    max_iters = len(train_dataloader)
    for epoch in range(start_epoch, stop_epoch):
        print(f"Epoch {epoch + 1}/{stop_epoch}", flush=True)
        total_loss = mean_loss = 0

        # Set model to training mode.
        model.train()
        with ProgressBar(
            train_dataloader, total=max_iters, desc="train"
        ) as pbar:
            for idx, (mixture, target) in enumerate(pbar):
                with autocast(enabled=model.use_amp):

                    # Process data, run forward pass.
                    model.set_data(mixture, target)
                    model.forward()

                    # Calculate mini-batch loss.
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

                # Run backprop.
                model.backward()
                # Visualize gradients.
                callback.on_loss_end(global_step=global_step)
                # Update model parameters.
                model.optimizer_step()
                # Write/display iteration loss.
                callback.on_iteration_end(global_step=global_step)
                global_step += 1

        # Store epoch-average training loss.
        model.train_losses.append(mean_loss)
        # Run validation loop, write/display train and val losses.
        callback.on_epoch_end(epoch=epoch)
        # Only save model if validation loss decreases.
        if model.is_best_model:
            model.save_all(
                global_step=epoch,
                model=True,
                optim=True,
                scheduler=True,
                grad_scaler=model.use_amp,
            )

        # Decrease lr if necessary.
        stop_early = model.scheduler_step()
        if stop_early:
            print("No improvement. Stopping training early...")
            break

