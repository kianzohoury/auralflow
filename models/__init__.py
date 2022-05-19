import torch


from . architectures import (
    SpectrogramNetSimple, SpectrogramNetLSTM, SpectrogramNetVAE
)
from . base import SeparationModel
from . mask_model import SpectrogramMaskModel
from pathlib import Path


__all__ = [
    "SpectrogramNetSimple",
    "SpectrogramNetLSTM",
    "SpectrogramNetVAE",
    "SeparationModel",
    "SpectrogramMaskModel",
    "save_object",
    "load_object",
    "create_model",
    "setup_model"
]


def create_model(configuration: dict) -> SeparationModel:
    """Creates a new instance of a model with its given configuration."""
    separation_task = configuration["model_params"]["separation_task"]
    if separation_task == "mask":
        base_class = SpectrogramMaskModel
    else:
        base_class = lambda x: x
        pass

    model = base_class(configuration)
    return model


def setup_model(model: SeparationModel):
    if model.training_mode:
        last_epoch = model.training_params["last_epoch"]
        if model.training_params["last_epoch"] >= 0:
            model.load_model(global_step=last_epoch)
            model.load_optim(global_step=last_epoch)
            model.load_scheduler(global_step=last_epoch)
            if model.training_params["use_mixed_precision"]:
                model.load_grad_scaler(global_step=last_epoch)
        else:
            Path(model.checkpoint_path).mkdir(exist_ok=True)
            pass

        # load model, load scheduler, load optimizer, load scaler,

    else:
        pass



def add_checkpoint_tag(filename: str, obj_name: str, global_step: int) -> str:
    """Attaches a suffix to the checkpoint filename depending on the object."""
    if obj_name == "model":
        filename += f"_{global_step}.pth"
    elif obj_name == "optimizer":
        filename += f"_optimizer_{global_step}.pth"
    elif obj_name == "scheduler":
        filename += f"_scheduler_{global_step}.pth"
    elif obj_name == "grad_scaler":
        filename += f"_autocast_{global_step}.pth"
    return filename


def save_object(
    model_wrapper, obj_name: str, global_step: int, silent: bool = True
) -> None:
    """Saves object state as .pth file under the checkpoint directory."""
    filename = f"{model_wrapper.checkpoint_path}/{model_wrapper.model_name}"

    # Get object-specific filename.
    filename = add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )

    if hasattr(model_wrapper, obj_name):
        # Retrieve object's state.
        if obj_name == "model":
            state_dict = getattr(model_wrapper, obj_name).cpu().state_dict()
            # Transfer model back to GPU if applicable.
            model_wrapper.model.to(model_wrapper.device)
        else:
            state_dict = getattr(model_wrapper, obj_name).state_dict()
        try:
            # Save object's state to filename.
            torch.save(state_dict, f=filename)
            if not silent:
                print(f"Successfully saved {obj_name}.")
        except OSError as error:
            print(f"Failed to save {obj_name} state.")
            raise error


def load_object(model_wrapper, obj_name: str, global_step: int) -> None:
    """Loads object and attaches it to model_wrapper and its device."""
    filename = f"{model_wrapper.checkpoint_path}/{model_wrapper.model_name}"

    # Get object-specific filename.
    filename = add_checkpoint_tag(
        filename=filename, obj_name=obj_name, global_step=global_step
    )

    try:
        # Try to read object state from the given file.
        state_dict = torch.load(filename, map_location=model_wrapper.device)
    except (OSError, FileNotFoundError) as error:
        print(f"Failed to load {obj_name} state.")
        raise error
    if hasattr(model_wrapper, obj_name):
        # Load state into object.
        getattr(model_wrapper, obj_name).load_state_dict(state_dict)
        print(f"Loaded {obj_name} successfully.")

