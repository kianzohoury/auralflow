# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import torch

from auralflow.models import SeparationModel
from auralflow.visualizer import ProgressBar
from auralflow.utils import save_all, save_config
from .callbacks import TrainingCallback
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


def run_training(
    model: SeparationModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    callback: TrainingCallback,
) -> None:
    """Runs training loop.

    Args:
        model (SeparationModel): Separation model.
        train_dataloader (DataLoader): Training set dataloader.
        val_dataloader (DataLoader): Validation set dataloader.
        callback (TrainingCallback): Training callbacks.
    """
    start_epoch = model.training_params["last_epoch"] + 1
    stop_epoch = start_epoch + model.training_params["max_epochs"]
    global_step = model.training_params["global_step"] + 1
    max_iters = len(train_dataloader)

    for epoch in range(start_epoch, stop_epoch):
        print(f"Epoch {epoch}/{stop_epoch - 1}", flush=True)
        total_loss = mean_loss = 0

        # Set model to training mode.
        model.train()
        with ProgressBar(
            train_dataloader, total=max_iters, desc="train"
        ) as pbar:
            for idx, (mixture, target) in enumerate(pbar):
                with autocast(enabled=model._use_amp):

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
                # Update global step count.
                model.training_params["global_step"] = global_step
                global_step += 1

        # Store epoch-average training loss.
        model.train_losses.append(mean_loss)

        # Run validation loop.
        run_validation(model=model, val_dataloader=val_dataloader)

        # Write/display train and val losses.
        callback.on_epoch_end(epoch=epoch, *next(iter(val_dataloader)))

        # Update epoch count.
        model.training_params["last_epoch"] = epoch

        # Only save model if validation loss decreases.
        if model._is_best_model:
            model.training_params["best_epoch"] = epoch
            save_all(
                model=model,
                global_step=epoch,
                save_model=True,
                save_optim=True,
                save_scheduler=True,
                save_grad_scaler=True
            )

        # Decrease lr if necessary.
        stop_early = model.scheduler_step()
        if stop_early:
            print("No improvement. Stopping training early...")
            break

        # Save updated config file.
        save_config(
            config=model.config,
            save_filepath=model.model_params["save_dir"]
        )


def run_validation(model: SeparationModel, val_dataloader: DataLoader) -> None:
    """Runs validation loop.

    Args:
        model (SeparationModel): Separation model.
        val_dataloader (DataLoader): Validation set dataloader.
    """
    max_iters = len(val_dataloader)

    # Set model to validation mode.
    model.eval()
    with ProgressBar(val_dataloader, total=max_iters, desc="valid") as pbar:
        total_loss = mean_loss = 0
        for idx, (mixture, target) in enumerate(pbar):
            with autocast(enabled=model._use_amp):
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
