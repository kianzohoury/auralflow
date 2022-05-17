from torch.utils.tensorboard import SummaryWriter
from .visualizer import Visualizer
from .progress import ProgressBar


__all__ = ["Visualizer", "config_visualizer", "ProgressBar"]


def config_visualizer(config: dict, writer: SummaryWriter) -> Visualizer:
    """Initializes and returns a visualizer object."""
    visualizer_params = config["visualizer_params"]
    dataset_params = config["dataset_params"]
    viz = Visualizer(
        writer=writer,
        save_dir=visualizer_params["logs_path"] + "/images",
        view_images=visualizer_params["view_images"],
        view_gradients=visualizer_params["view_gradients"],
        play_audio=visualizer_params["play_audio"],
        num_images=visualizer_params["num_images"],
        save_image=visualizer_params["save_images"],
        save_audio=visualizer_params["save_audio"],
        save_freq=visualizer_params["save_frequency"],
        sample_rate=dataset_params["sample_rate"],
    )
    return viz
