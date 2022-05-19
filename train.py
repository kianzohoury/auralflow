# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import sys
from argparse import ArgumentParser
from datasets import create_audio_dataset, load_dataset
from losses import SeparationEvaluator
from models import create_model, setup_model
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from utils import load_config
from validate import cross_validate
from visualizer import ProgressBar, config_visualizer
import torch.backends.cudnn


def main(config_filepath: str):
    """Runs training script given a configuration file."""
    print(torch.backends.cudnn.benchmark)

    # Load configuration file.
    print("-" * 79 + "\nReading configuration file...")
    configuration = load_config(config_filepath)
    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    visualizer_params = configuration["visualizer_params"]
    print("Successful.")

    # Load training set into memory.
    print("-" * 79 + "\nLoading training data...")
    train_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="train",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=dataset_params["max_num_samples"],
        max_num_tracks=dataset_params["max_num_tracks"],
        sample_rate=dataset_params["sample_rate"],
        mono=dataset_params["num_channels"],
    )

    # Load validation set into memory.
    val_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="val",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=dataset_params["max_num_samples"],
        max_num_tracks=dataset_params["max_num_tracks"],
        mono=dataset_params["num_channels"],
    )

    # Load data into dataloaders.
    train_dataloader = load_dataset(train_dataset, training_params)
    val_dataloader = load_dataset(val_dataset, training_params)
    print(
        f"Successful. Loaded {len(train_dataset)} training and "
        f"{len(val_dataset)} validation samples of length "
        f"{dataset_params['sample_length']}s."
    )

    # Load model. Setup restores previous state if resuming training.
    print("-" * 79 + "\nLoading model...")
    model = create_model(configuration)
    setup_model(model)
    print("Successful.")

    # Initialize summary writer and visualizer.
    print("-" * 79 + "\nLoading visualization tools...")
    writer = SummaryWriter(log_dir=visualizer_params["logs_path"])
    visualizer = config_visualizer(config=configuration, writer=writer)
    print("Successful.")

    # Number of epochs to train.
    start_epoch = training_params["last_epoch"] + 1
    stop_epoch = start_epoch + training_params["max_epochs"]
    global_step = configuration["training_params"]["global_step"]
    max_iters = len(train_dataloader)

    print("Configuration complete. Starting training...\n" + "-" * 79)
    for epoch in range(start_epoch, stop_epoch):
        print(f"Epoch {epoch}/{stop_epoch}", flush=True)
        total_loss = 0
        train_loss = []
        model.train()
        with ProgressBar(train_dataloader, total=max_iters) as pbar:
            for idx, (mixture, target) in enumerate(pbar):
                # Cast precision if necessary to increase training speed.
                # with autocast():

                # Process data, run forward pass.
                model.set_data(mixture, target)
                model.forward()

                # Calculate mini-batch loss and run backprop.
                batch_loss = model.compute_loss()

                total_loss += batch_loss
                train_loss.append(batch_loss)

                model.backward()

                # Mid-epoch callback.
                model.mid_epoch_callback(visualizer=visualizer, epoch=epoch)

                # Update model parameters.
                model.optimizer_step()
                global_step += 1

                # Display and log the loss.
                pbar.set_postfix({"train_loss": batch_loss})
                writer.add_scalars(
                    "Loss/train/iter",
                    {f"{model.criterion.__class__.__name__}": batch_loss},
                    global_step,
                )

        avg_loss = sum(train_loss) / len(train_loss)
        pbar.set_postfix({"train_loss": round(batch_loss, 6), "avg": avg_loss})
        pbar.set_postfix({"loss": round(avg_loss, 6)})
        # Store epoch-average loss.
        model.train_losses.append(avg_loss)
        writer.add_scalars(
            "Loss/train/epoch",
            {f"{model.criterion.__class__.__name__}": avg_loss},
            epoch,
        )

        # Validate updated model.
        cross_validate(
            model=model,
            val_dataloader=val_dataloader,
        )

        # metrics = evaluator.get_metrics(*next(iter(val_dataloader)))

        # writer.add_scalars("eval_metrics", metrics, global_step=epoch)

        print("avg train loss:", model.train_losses[-1])
        print("avg valid loss:", model.val_losses[-1])
        # SeparationEvaluator.print_metrics(metrics)
        print("-" * 79)

        # Decrease lr if necessary.
        stop_early = model.scheduler_step()

        # Log validation loss.
        writer.add_scalars(
            "Loss/val/epoch",
            {f"{model.criterion.__class__.__name__}": model.val_losses[-1]},
            epoch,
        )

        writer.add_scalars(
            "Loss/comparison",
            {"train": model.train_losses[-1], "val": model.val_losses[-1]},
            epoch,
        )

        if stop_early:
            print("No improvement. Stopping training early...")
            break

        # Only save model if validation loss decreases.
        if model.is_best_model:
            model.save_model(global_step=epoch, silent=model.silent_checkpoint)
            model.save_optim(global_step=epoch, silent=model.silent_checkpoint)
            model.save_scheduler(
                global_step=epoch, silent=model.silent_checkpoint
            )
            model.save_grad_scaler(
                global_step=epoch, silent=model.silent_checkpoint
            )

        # Post-epoch callback.
        model.post_epoch_callback(
            *next(iter(val_dataloader)), visualizer=visualizer, epoch=epoch
        )

    writer.close()
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Model training script.")
    parser.add_argument(
        "config_filepath", type=str, help="Path to a configuration file."
    )
    args = parser.parse_args()
    main(args.config_filepath)
