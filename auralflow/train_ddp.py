# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import auralflow.__main__ as main_script
import torch.cuda
import shutil
from auralflow import configurations
from auralflow import parse
import sys
import os
from pathlib import Path
import importlib
import json

# importlib.reload(main_script)




def model_fn(model_dir):
    print(f"Building model...")
    model = configurations._build_model(
        model_config=model_config,
        device=training_config.device
    )
    print("  Successful.")


if __name__ == "__main__":
    print(os.environ["SM_HPS"])
    
    config_params = json.loads(os.environ["SM_HPS"])
    print(config_params)
    config_params = {
        key.replace("-", "_"): val for (key, val) in config_params.items()
    }

    # Create model configuration from args.
    model_config = configurations._create_model_config(
        model_type=config_params.pop("model_type"),
        targets=config_params["targets"],
        **config_params
    )

    # Save the model configuration file within the specified save dir.
    save_dir = Path(os.environ["SM_OUTPUT_DATA_DIR"], config_params["save"])
    if save_dir.is_dir():
        shutil.rmtree(str(save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    model_config.save(filepath=str(save_dir.joinpath("model.json")))

    # Display model config as a table.
    if config_params[display]:
        print(model_config)

    print(f"Model configuration successfully saved to {str(save_dir)}.")
    #
    # # Create loss criterion configuration.
    # if isinstance(model_config, configurations.SpecModelConfig):
    #     input_type = "spectrogram"
    # else:
    #     input_type = "audio"
    # criterion_config = configurations.CriterionConfig.from_dict(
    #     input_type=input_type, **args.__dict__
    # )
    #
    # # Set logging and image directories.
    # if not args.tensorboard:
    #     logging_dir = image_dir = None
    # else:
    #     # Assign to default directories.
    #     logging_dir = str(save_dir.joinpath("runs"))
    #     image_dir = str(save_dir)
    #
    # # Create visualization configuration.
    # visuals_config = configurations.VisualsConfig.from_dict(
    #     logging_dir=logging_dir,
    #     image_dir=image_dir,
    #     **args.__dict__
    # )
    #
    # # Create trainer configuration.
    # training_config = configurations.TrainingConfig.from_dict(
    #     criterion_config=criterion_config,
    #     visuals_config=visuals_config,
    #     checkpoint=str(save_dir.joinpath("checkpoint.pth")),
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     **args.__dict__
    # )
    #
    # # Save the training configuration in the same directory.
    # training_config.save(filepath=str(save_dir.joinpath("trainer.json")))
    #
    # if not args.resume:
    #     # Delete existing metadata files.
    #     for metadata_file in list(save_dir.glob("*.pickle")):
    #         metadata_file.unlink()
    #     # Delete existing checkpoint.
    #     prev_checkpoint = save_dir.joinpath("checkpoint.pth")
    #     if prev_checkpoint.exists():
    #         prev_checkpoint.unlink()
    #
    # if args.display:
    #     print(training_config)
    #
    # # Run training.
    # train.main(
    #     model_config=model_config,
    #     save_dir=str(save_dir),
    #     training_config=training_config,
    #     dataset_path=args.dataset_path,
    #     max_num_tracks=args.max_tracks,
    #     max_num_samples=args.max_samples,
    #     resume=args.resume
    # )
