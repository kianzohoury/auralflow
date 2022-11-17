# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import auralflow.__main__ as main_script
from auralflow import configurations
from auralflow import parse
import sys
import os
from pathlib import Path
import importlib

# importlib.reload(main_script)




def model_fn(model_dir):
    print(f"Building model...")
    model = configurations._build_model(
        model_config=model_config,
        device=training_config.device
    )
    print("  Successful.")



if __name__ == "__main__":
    
    # manually hack arguments for now
    print(sys.argv)

        # Create model configuration from args.
        model_config = configurations._create_model_config(
            model_type=args.__dict__.pop("model_type"),
            targets=parse.parse_targets(args),
            **args.__dict__
        )

        # Save the model configuration file within the specified save dir.
        save_dir = Path(args.save)
        if save_dir.is_dir():
            shutil.rmtree(str(save_dir))
        save_dir.mkdir(parents=True, exist_ok=True)
        model_config.save(filepath=str(save_dir.joinpath("model.json")))

        # Display model config as a table.
        if args.display:
            print(model_config)

        print(f"Model configuration successfully saved to {str(save_dir)}.")
    
#     i = 0
#     while i < len(sys.argv):
#         if sys.argv[i] == "--train":
#             break
#         i += 1 
        
#     sys.argv = ['/home/ec2-user/SageMaker/auralflow/auralflow/__main__.py'] + sys.argv[1:i]
    
    # config model
#     os.system(f"python3 {file_path} " + " ".join(sys.argv[1:i]))
#     sys.argv = [file_path] + sys.argv[:i]
    
#     print(sys.argv)
#     main_script.main()
    
#     # train model
#     sys.argv = [file_path] + sys.argv[i:]
#     auralflow.__main__.main()
