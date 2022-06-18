# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from auralflow.build import build_model, setup_model
from auralflow.datasets import create_audio_dataset, load_dataset
from torch.utils.tensorboard import SummaryWriter
from auralflow.trainer import run_training
from auralflow.trainer.callbacks import TrainingCallback
from auralflow.utils import load_config, save_config
from auralflow.visualizer import config_visualizer


def main(config_filepath: str):
    """Runs training script given a configuration file."""

    # Load configuration file.
    print("Reading configuration file...")
    configuration = load_config(config_filepath)
    model_params = configuration["model_params"]
    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    print("  Successful.")

    # Load training set into memory.
    print("Loading training data...")
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
        f"  Successful.\nLoaded {len(train_dataset)} training and "
        f"{len(val_dataset)} validation samples of length "
        f"{dataset_params['sample_length']}s."
    )

    # Load model. Setup restores previous state if resuming training.
    print(f"Loading {model_params['model_name']}...")
    model = build_model(configuration)
    model = setup_model(model)

    # Initialize summary writer and visualizer.
    print("Loading visualization tools...")
    writer = SummaryWriter(log_dir=model_params["save_dir"] + "/runs")
    visualizer = config_visualizer(config=configuration, writer=writer)
    print("  Successful.")

    # Create a callback object.
    callback = TrainingCallback(
        model=model, writer=writer, visualizer=visualizer, call_metrics=True
    )

    # Run training loop.
    print("Starting training...\n" + "-" * 79)
    run_training(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callback=callback,
    )

    # Save last checkpoint to resume training later.
    model.save(
        global_step=model.training_params["last_epoch"],
        model=True,
        optim=True,
        scheduler=True,
        grad_scaler=model.use_amp,
    )

    writer.close()
    print("Finished training.")

    # Save updated config file.
    save_config(
        config=configuration,
        save_filepath=model_params["save_dir"] + "/config.json"
    )
