# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from argparse import ArgumentParser
from auralflow.datasets import create_audio_dataset, load_dataset
from auralflow.models import create_model, setup_model
from torch.utils.tensorboard import SummaryWriter
from auralflow.trainer import run_training
from auralflow.trainer.callbacks import TrainingCallback
from auralflow.utils import load_config, save_config
from auralflow.visualizer import config_visualizer


def main(config_filepath: str):
    """Runs training script given a configuration file."""

    # Load configuration file.
    print("-" * 79 + "\nReading configuration file...")
    configuration = load_config(config_filepath)
    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    print("Successful.")

    # Load training set into memory.
    print("-" * 79 + "\nLoading training data...")
    train_dataset = create_audio_dataset(
        dataset_path=dataset_params["dataset_path"],
        split="train",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=dataset_params["max_num_samples"],
        max_num_tracks=dataset_params["max_num_tracks"],
        sample_rate=dataset_params["sample_rate"],
        mono=dataset_params["num_channels"] == 1,
    )

    # Load validation set into memory.
    val_dataset = create_audio_dataset(
        dataset_path=dataset_params["dataset_path"],
        split="val",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=dataset_params["max_num_samples"],
        max_num_tracks=dataset_params["max_num_tracks"],
        mono=dataset_params["num_channels"] == 1,
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
    model = setup_model(model)
    print("Successful.")

    # Initialize summary writer and visualizer.
    print("-" * 79 + "\nLoading visualization tools...")
    writer = SummaryWriter(
        log_dir=configuration["model_params"]["save_dir"] + "/runs"
    )
    visualizer = config_visualizer(config=configuration, writer=writer)
    print("Successful.")

    # Number of epochs to train.
    start_epoch = training_params["last_epoch"] + 1
    stop_epoch = start_epoch + training_params["max_epochs"]
    global_step = configuration["training_params"]["global_step"] + 1

    # Create a callback object.
    callback = TrainingCallback(
        model=model,
        writer=writer,
        visualizer=visualizer,
        call_metrics=True
    )

    print("Configuration complete. Starting training...\n" + "-" * 79)
    epoch, global_step = run_training(
        model=model,
        start_epoch=start_epoch,
        stop_epoch=stop_epoch,
        global_step=global_step,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callback=callback,
    )

    writer.close()
    print("Finished.")

    training_params["last_epoch"] = stop_epoch
    training_params["global_step"] = global_step
    save_config(config=configuration, save_filepath=config_filepath)

