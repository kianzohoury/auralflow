import os
import subprocess
import sys
import threading
import math
import torch
import torch.nn as nn
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import autocast

from datasets import create_audio_dataset, load_dataset
from models import create_model
from utils import load_config
from validate import cross_validate
from visualizer.progress import ProgressBar
from visualizer import log_gradients


def main(config_filepath: str):
    """Runs training script given a configuration file."""

    print("-" * 79 + "\nReading configuration file...")
    configuration = load_config(config_filepath)
    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    dataloader_params = dataset_params["loader_params"]
    visualizer_params = configuration["visualizer_params"]
    print("Done.")

    print("-" * 79 + "\nFetching dataset...")
    train_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="train",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=int(2e2),
    )
    val_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="val",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=int(2e2),
    )

    train_dataloader = load_dataset(
        dataset=train_dataset, loader_params=dataloader_params
    )
    val_dataloader = load_dataset(
        dataset=val_dataset, loader_params=dataloader_params
    )
    print(
        f"Done. Loaded {len(train_dataset)} training and {len(val_dataset)}"
        f" validation samples of length {dataset_params['sample_length']}s."
    )

    print("-" * 79 + "\nLoading model...")
    model = create_model(configuration)
    # model.setup()
    print("Done.")

    # writer_process = Process(
    #     target=run_tensorboard, args=(visualizer_params["logs_path"],)
    # )

    # writer_process.start()

    writer = SummaryWriter(log_dir=visualizer_params["logs_path"])

    current_epoch = training_params["last_epoch"] + 1
    stop_epoch = current_epoch + training_params["max_epochs"]
    global_step = configuration["training_params"]["global_step"]
    # max_iters = dataloader_params["max_iterations"]
    max_iters = len(train_dataloader)
    save_freq = training_params["checkpoint_freq"]

    print("-" * 79 + "\nStarting training...")
    for epoch in range(current_epoch, stop_epoch):
        print(f"Epoch [{epoch}/{stop_epoch}]", flush=True)
        total_loss = 0
        model.train()
        with ProgressBar(train_dataloader, max_iters, desc="train:") as pbar:
            for idx, (mixture, target) in enumerate(pbar):
                # Cast precision if necessary to increase training speed.
                # with autocast(device_type=model.device):
                model.set_data(mixture, target)
                # print(model.mixtures.shape)
                model.forward()

                # Compute batch-wise loss.
                batch_loss = model.get_loss()
                model.backward()

                log_gradients(
                    model=model.model, writer=writer, global_step=epoch
                )

                # nn.utils.clip_grad_norm_(
                #     model.model.parameters(), max_norm=1.0
                # )

                # Update model parameters.
                model.optimizer_step()
                global_step += 1
                # Accumulate loss.
                total_loss += model.batch_loss

                # Display and log the loss.
                pbar.set_postfix({"loss": batch_loss})
                writer.add_scalars(
                    "Loss/train",
                    {"train": batch_loss},
                    global_step,
                )

                # Break if looped max_iters times.
                if idx == max_iters:
                    pbar.clear()
                    break

        # Store epoch-average loss.
        model.train_losses.append(total_loss / max_iters)

        # Validate updated model.
        cross_validate(
            model=model,
            val_dataloader=val_dataloader,
        )

        # Decrease lr if scheduler determines so.
        model.scheduler_step()

        # Log validation loss.
        writer.add_scalars(
            "Loss/val",
            {"val": model.val_losses[-1]},
            epoch,
        )

        # Only save the best model.
        is_best = model.val_losses[-1] < min(model.val_losses)
        if is_best:
            model.save_model(global_step=epoch)
            model.save_optim(global_step=epoch)

        # Stop training if stop patience runs out before improvement.
        if model.stop_early():
            print("Stopping training early...")
            break

        model.post_epoch_callback(mixture, target, writer, epoch)

    writer.close()
    print("-" * 90)
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Model training script.")
    parser.add_argument(
        "config_filepath", type=str, help="Path to a configuration file."
    )
    args = parser.parse_args()
    # run_tensorboard("logs")
    # writer_process = Process(target=run_tensorboard, args=("runs",))
    main(args.config_filepath)

    # writer_process.start()
    # main_process = Process(target=main, args=(args.config_filepath,))
    # main_process.start()
