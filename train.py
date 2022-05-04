import time
from argparse import ArgumentParser

import torch
import numpy as np

from datasets import create_audio_folder, load_dataset, create_audio_dataset
from models import create_model
from utils import load_config
from utils.progress_bar import ProgressBar
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from torch.profiler import profile, record_function
from torch.profiler.profiler import ProfilerActivity


def main(config_filepath: str):

    print("Reading configuration file...")
    configuration = load_config(config_filepath)

    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    loader_params = dataset_params["loader_params"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    # train_dataset = create_audio_folder(
    #     dataset_params=dataset_params, subset="train"
    # )
    train_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="train",
        targets=["vocals"],
        chunk_size=3,
        num_chunks=int(1e6),
    )
    # val_dataset = train_dataset.split(val_split=dataset_params["val_split"])
    # train_dataloader = load_dataset(
    #     dataset=train_dataset, loader_params=dataset_params["loader_params"]
    # )
    train_dataloader = DataLoader(
        train_dataset, num_workers=8, pin_memory=True,
        persistent_workers=True, batch_size=64, prefetch_factor=4, shuffle=True)
    # val_dataloader = load_dataset(
    #     dataset=val_dataset, loader_params=dataset_params["loader_params"]
    # )
    print("Loading complete.")

    model = create_model(configuration)
    # model.setup()

    print("=" * 95)
    print("Training session started...")
    print("=" * 95)

    current_epoch = training_params["last_epoch"] + 1
    stop_epoch = current_epoch + training_params["max_epochs"]
    # max_iters_per_epoch = loader_params["max_iterations"]
    global_step = configuration["training_params"]["global_step"]

    writer = SummaryWriter("runs_vocals_1")
    iters = 50
    model.train()
    for epoch in range(current_epoch, stop_epoch):
        total_loss = 0
        with ProgressBar(train_dataloader, iters) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{stop_epoch}]")
            for index, (mixture, target) in enumerate(pbar):
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_inference"):
            
                model.set_data(mixture, target)
                model.forward()
                model.backward()
                model.optimizer_step()

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                writer.add_scalars("Loss/train", {"batch_64_lr_0005_VAE_1024": model.train_losses[-1]}, global_step)
                pbar.set_postfix({"avg_loss": model.train_losses[-1]})
                total_loss += model.train_losses[-1]

                global_step += 1
                # start = time.time()

                # break after seeing max_iter * batch_size samples
                if index >= iters:
                    pbar.set_postfix({"avg_loss": total_loss / iters})
                    pbar.clear()
                    break
        pbar.set_postfix({"avg_loss": total_loss / iters})
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        # epoch_losses.append(total_loss / max_iters)

        # if epoch % 10 == 0:
        #     torch.save(training_session, training_session["latest_checkpoint"])
        #
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     training_session["best_val_loss"] = best_val_loss
        #     stop_counter = 0
        #     torch.save(training_session, training_session["best_checkpoint"])
        # elif stop_counter < patience:
        #     stop_counter += 1
        #     epochs_left = patience - stop_counter + 1
        #     if epoch < epochs:
        #         print("=" * 90)
        #         print(
        #             f"Early Stopping: {epochs_left} epochs left if no "
        #             "improvement is made."
        #         )
        #         print("=" * 90)
        # else:
        #     break

    print("=" * 90 + "\nTraining finished.")


if __name__ == "__main__":

    parser = ArgumentParser(description="Model training script.")
    parser.add_argument(
        "config_filepath", type=str, help="Path to a configuration file."
    )
    args = parser.parse_args()
    main(args.config_filepath)

