# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

from auralflow.datasets import create_audio_dataset
from auralflow.trainer import _DefaultModelTrainer
from auralflow import configurations
from torch.utils.data import DataLoader
from pathlib import Path


def main(
    model_config: configurations.AudioModelConfig,
    training_config: configurations.TrainingConfig,
    save_dir: str,
    dataset_path: str,
    max_num_tracks: int = 80,
    max_num_samples: int = 10000,
    resume: bool = True
) -> None:
    """Trains a model given a valid path to a configuration file.

    The configuration file must have valid keys and values specifying how to
    build the model, how to train it, how to construct the dataset and how to
    visualize training (e.g. tensorboard, saving files locally).

    Args:
        model_config (AudioModelConfig): Model configuration.
        training_config (TrainingConfig): Training configuration.
        dataset_path (str): Path to the dataset.
        max_num_tracks (int): Max number of tracks to load into memory.
            Default: 80.
        max_num_samples (int): Max number of resampled chunks from the pool
            of tracks. Default: 10000.
    """
    print("Loading training data...")
    train_dataset_metadata_path = str(Path(save_dir).joinpath(
        ".train_dataset_metadata.pickle")
    )
    val_dataset_metadata_path = str(Path(save_dir).joinpath(
        ".val_dataset_metadata.pickle")
    )
    
    train_dataset = create_audio_dataset(
        dataset_path=dataset_path,
        split="train",
        targets=model_config.targets,
        chunk_size=model_config.sample_length,
        max_num_tracks=max_num_tracks,
        num_chunks=max_num_samples,
        sample_rate=model_config.sample_rate,
        mono=model_config.num_channels == 1,
        metadata_path=train_dataset_metadata_path
    )

    # Load validation set.
    val_dataset = create_audio_dataset(
        dataset_path=dataset_path,
        split="val",
        targets=model_config.targets,
        chunk_size=model_config.sample_length,
        max_num_tracks=max_num_tracks,
        num_chunks=max_num_samples,
        sample_rate=model_config.sample_rate,
        mono=model_config.num_channels == 1,
        metadata_path=val_dataset_metadata_path
    )

    # Load data into dataloaders.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
        prefetch_factor=training_config.pre_fetch,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
        prefetch_factor=training_config.pre_fetch,
        shuffle=True
    )
    print(
        f"  Successful. Loaded {len(train_dataset)} training and "
        f"{len(val_dataset)} validation samples of length "
        f"{model_config.sample_length}s."
    )

    # Build and initialize the source separation model.
    print(f"Building model...")
    model = configurations._build_model(
        model_config=model_config,
        device=training_config.device
    )
    print("  Successful.")

    print(f"Loading trainer...")
    # Get criterion.
    criterion = configurations._get_loss_criterion(
        criterion_config=training_config.criterion_config
    )
    # Initialize trainer.
    trainer = _DefaultModelTrainer(
        model=model,
        criterion=criterion,
        mixed_precision=training_config.use_amp,
        scale_grad=training_config.scale_grad,
        clip_grad=training_config.clip_grad,
        checkpoint=training_config.checkpoint,
        logging_dir=training_config.visuals_config.logging_dir,
        resume=resume,
        device=training_config.device,
        lr=[training_config.lr, training_config.lr_lstm],
        init_scale=training_config.init_scale,
        max_grad_norm=training_config.max_grad_norm,
        max_plateaus=training_config.max_plateaus,
        stop_patience=training_config.stop_patience,
        min_delta=training_config.min_delta,
        view_norm=training_config.visuals_config.view_norm,
        view_epoch=training_config.visuals_config.view_epoch,
        view_iter=training_config.visuals_config.view_iter,
        view_grad=training_config.visuals_config.view_grad,
        view_weights=training_config.visuals_config.view_weights,
        view_spec=training_config.visuals_config.view_spec,
        view_wave=training_config.visuals_config.view_wave,
        play_estimate=training_config.visuals_config.play_estimate,
        play_residual=training_config.visuals_config.play_residual,
        image_dir=training_config.visuals_config.image_dir,
        image_freq=training_config.visuals_config.image_freq,
        silent=training_config.visuals_config.silent
    )
    print("  Successful.")

    # Run training.
    print("Starting training...\n" + "-" * 79)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=training_config.max_epochs
    )
    print("Finished training.")
