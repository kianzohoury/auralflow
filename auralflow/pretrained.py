# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

import gdown
import shutil
import zipfile

from auralflow.models.mask_model import SeparationModel
from .configurations import _build_model, _load_model_config
from pathlib import Path
from typing import Optional

# temporarily place links here
_model_links = {
    "SpectrogramNetLSTM_vocals": "109-W23fjoPN1EyqkiStEgr0dOXLVsrBv"
}


def load(
    model: str, target: str, save_dir: Optional[str] = None, device: str = 'cpu'
) -> SeparationModel:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_zip = gdown.download(
        id=_model_links[f"{model}_{target}"], output=save_dir + "/model.zip"
    )
    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        for file in zip_ref.infolist()[1:]:
            file_name = Path(file.filename).name
            dest = f"{save_dir}/{file_name}"
            # Path(dest).parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(file.filename, mode="r") as temp_file:
                with open(dest, mode="wb") as dest_file:
                    try:
                        shutil.copyfileobj(temp_file, dest_file)
                    except zipfile.error as e:
                        raise e
    model_config = _load_model_config(
        filepath=f"{save_dir}/model.json"
    )

    model = _build_model(model_config, device=device)
    return model
