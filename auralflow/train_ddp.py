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

    
