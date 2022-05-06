import os
import threading
from argparse import ArgumentParser
from multiprocessing import Process
import inspect
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import create_audio_dataset
from models import create_model
from utils import load_config
from visualizer.progress import ProgressBar
import subprocess
from pathlib import Path
import tensorboard
import sys


def run_tensorboard(log_dir):
    # os.chdir(Path(__file__).parent / "runs")
    # [f"python -m tensorboard.main --logdir {log_dir}"]
    
    tb_path = ["tensorboard", "--logdir", 'logs']
    # command = ["python", "-m", "tensorboard.main", "--logdir", log_dir]
    command = tb_path
    print(command)
    
    # tb_thread = threading.Thread(
    #     target=lambda: subprocess.Popen(command, shell=False, cwd=os.getcwd()).communicate(),
    #     daemon=True,
    # )
    # tb_thread.start()
    print(subprocess.Popen(command, shell=False, cwd=os.getcwd()).communicate())
    print(subprocess.check_output(command), shell=True)
    sys.exit(1)
    


def main(config_filepath: str):
    print("=" * 95)
    print("Reading configuration file...")
    configuration = load_config(config_filepath)

    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    loader_params = dataset_params["loader_params"]
    visualizer_params = configuration["visualizer_params"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print("Loading training set...")
    train_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="train",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=int(1e4),
    )
    print("Completed.")
    print("=" * 95)
    print("Loading validation set...")
    val_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="val",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=int(1e4),
    )
    print("Completed.")
    print("=" * 95)

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=64,
        prefetch_factor=4,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=64,
        prefetch_factor=4,
        shuffle=True,
    )
    # val_dataloader = load_dataset(
    #     dataset=val_dataset, loader_params=dataset_params["loader_params"]
    # )

    print("Loading model...")
    model = create_model(configuration)
    model.setup()
    print("Completed.")

    print(visualizer_params["logs_path"])

    writer_process = Process(
        target=run_tensorboard, args=(visualizer_params["logs_path"],)
    )

    writer_process.start()

    writer = SummaryWriter(log_dir="FUCK")

    print("=" * 95)
    print("Training is starting...")
    print("=" * 95)

    current_epoch = training_params["last_epoch"] + 1
    stop_epoch = current_epoch + training_params["max_epochs"]
    global_step = configuration["training_params"]["global_step"]
    max_iters = min(4, loader_params["max_iterations"])
    save_freq = training_params["checkpoint_freq"]
    val_step = 0

    model.train()
    for epoch in range(current_epoch, stop_epoch):
        total_loss = 0
        with ProgressBar(train_dataloader, max_iters) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{stop_epoch}]")
            for index, (mixture, target) in enumerate(pbar):

                model.set_data(mixture, target)
                model.forward()
                model.backward()
                model.optimizer_step()

                batch_loss = model.get_batch_loss()
                pbar.set_postfix({"loss": batch_loss})
                total_loss += batch_loss

                global_step += 1

                if index >= max_iters:
                    pbar.set_postfix({"avg_loss": total_loss / max_iters})
                    pbar.clear()
                    break

                writer.add_scalars(
                    "Loss/train",
                    {"batch_64_lr_0005_VAE_1024": batch_loss},
                    global_step,
                )

        model.train_losses.append(total_loss / max_iters)

        pbar.set_postfix({"avg_loss": total_loss / max_iters})

        if index % save_freq == 0:
            model.save_model(global_step=global_step)
            model.save_optim(global_step=global_step)
        # if model.stop_early():
        #     break

        model.post_epoch_callback(writer, epoch, val_dataloader, max_iters)

    writer.close()
    print("=" * 90 + "\nTraining is complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Model training script.")
    parser.add_argument(
        "config_filepath", type=str, help="Path to a configuration file."
    )
    args = parser.parse_args()
    writer_process = Process(
        target=run_tensorboard, args=("runs",)
    )

    writer_process.start()
    # main_process = Process(target=main, args=(args.config_filepath,))
    # main_process.start()
