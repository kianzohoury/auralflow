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

from datasets import create_audio_dataset
from models import create_model
from utils import load_config
from validate import cross_validate
from visualizer.progress import ProgressBar


def main(config_filepath: str):
    print("-" * 79)
    print("Reading configuration file...")
    configuration = load_config(config_filepath)

    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    loader_params = dataset_params["loader_params"]
    visualizer_params = configuration["visualizer_params"]

    print("Done.")
    print("-" * 79)
    print("Fetching dataset...")
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
    print("Done.")
    print("-" * 79)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=64,
        prefetch_factor=4,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=64,
        prefetch_factor=4,
        shuffle=True,
    )
    # val_dataloader = load_dataset(
    #     dataset=val_dataset, loader_params=dataset_params["loader_params"]
    # )

    print("Loading model...")
    model = create_model(configuration)
    # model.setup()
    print("Done.")

    # writer_process = Process(
    #     target=run_tensorboard, args=(visualizer_params["logs_path"],)
    # )

    # writer_process.start()

    writer = SummaryWriter(log_dir=visualizer_params["logs_path"])

    print("-" * 79)
    print("Starting training...")
    
    current_epoch = training_params["last_epoch"] + 1
    stop_epoch = current_epoch + training_params["max_epochs"]
    global_step = configuration["training_params"]["global_step"]
    max_iters = 60
    # max_iters = loader_params["max_iterations"]
    save_freq = training_params["checkpoint_freq"]

    for epoch in range(current_epoch, stop_epoch):
        print(f"Epoch [{epoch}/{stop_epoch}]", flush=True)
        total_loss = 0
        model.train()
        with ProgressBar(train_dataloader, max_iters) as pbar:
            pbar.set_description("train")
            for index, (mixture, target) in enumerate(pbar):

                with autocast(device_type="cuda"):
                    model.set_data(mixture, target)
                    model.forward()
                model.backward()

                # if (index + 1) % model.accum_steps == 0:
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
            epoch=epoch,
        )

        writer.add_scalars(
            "Loss/val",
            {"l1_kl": model.val_losses[-1]},
            epoch,
        )

        if index % save_freq == 0:
            model.save_model(global_step=global_step)
            model.save_optim(global_step=global_step)

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
