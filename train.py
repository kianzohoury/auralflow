import os
import subprocess
import sys
import threading
import math
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import autocast

from datasets import create_audio_dataset, load_dataset
from models import create_model
from utils import load_config
from validate import cross_validate
from visualizer.progress import ProgressBar


def main(config_filepath: str):
    """Runs training script given a configuration file."""

    print("-" * 79 + "\nReading configuration file...")
    configuration = load_config(config_filepath)
    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    dataloader_params = dataset_params["loader_params"]
    visualizer_params = configuration["visualizer_params"]
    print("Done.")

    print("-" * 79 + "Fetching dataset...")
    train_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="train",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=int(1e5),
    )
    val_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="val",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=int(1e4),
    )

    train_dataloader = load_dataset(
        dataset=train_dataset, loader_params=dataloader_params
    )
    val_dataloader = load_dataset(
        dataset=val_dataset, loader_params=dataloader_params
    )
    print(
        f"Done. Loaded {len(train_dataset)} train and {len(val_dataset)}"
        f" {dataset_params['sample_length']}s val audio chunks."
    )

    print("-" * 79 + "Loading model...")
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
    max_iters = dataloader_params["max_iterations"]
    save_freq = training_params["checkpoint_freq"]

    print("-" * 79 + "Starting training...")
    for epoch in range(current_epoch, stop_epoch):
        print(f"Epoch [{epoch}/{stop_epoch}]", flush=True)
        total_loss = 0
        model.train()
        with ProgressBar(train_dataloader, max_iters, desc="train") as pbar:
            for index, (mixture, target) in enumerate(pbar):
                # Cast precision if necessary to increase training speed.
                with autocast(device_type=model.device):
                    model.set_data(mixture, target)
                    model.forward()

                # Compute batch-wise loss.
                model.backward()
                # Update model parameters.
                model.optimizer_step()

                batch_loss = model.get_batch_loss()
                pbar.set_postfix({"loss": batch_loss})
                total_loss += batch_loss

                global_step += 1

                if index == max_iters:
                    pbar.clear()
                    break

                writer.add_scalars(
                    "Loss/train",
                    {"l1_kl": batch_loss},
                    global_step,
                )

        model.train_losses.append(total_loss / max_iters)

        cross_validate(
            model=model,
            writer=writer,
            val_dataloader=val_dataloader,
            max_iters=max_iters,
        )

        model.scheduler_step()

        writer.add_scalars(
            "Loss/val",
            {"l1_kl": model.val_losses[-1]},
            epoch,
        )

        if index % save_freq == 0:
            model.save_model(global_step=epoch)
            model.save_optim(global_step=epoch)

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
