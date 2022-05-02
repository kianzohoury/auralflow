from pickle import TRUE
import sys
import time
from typing_extensions import TypeVarTuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import torchaudio
import torchinfo
from utils import load_config
from torch.utils import tensorboard
from models import create_model
import inspect
from utils.utils import checkpoint_handler

from pathlib import Path

import config.utils
from trainer.trainer import cross_validate
from utils.progress_bar import ProgressBar
from torch.utils.data.dataloader import DataLoader
from audio_folder import AudioFolder, create_dataset, load_dataset, AudioDataset, StreamDataset
from argparse import ArgumentParser
import config.build


def main(config_filepath: str):

    configuration = load_config(config_filepath)
    training_mode = configuration["training_params"]["training_mode"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing dataset...")
    train_dataset = create_dataset(
        dataset_params=configuration["dataset_params"], subset="train"
    )
    val_dataset = train_dataset.split(
        configuration["dataset_params"]["val_split"]
    )

    train_dataloader = load_dataset(
        dataset=train_dataset, dataset_params=configuration["dataset_params"]
    )
    val_dataloader = load_dataset(
        dataset=val_dataset, dataset_params=configuration["dataset_params"]
    )
    print("Finished loading data.")

    model = create_model(configuration)
    # model.setup()

    print("=" * 95)
    print("Training session started...")
    print("=" * 95)

    current_epoch = configuration["training_params"]["last_epoch"] + 1
    max_epochs = configuration["training_params"]["max_epochs"]
    stop_epoch = current_epoch + max_epochs
    max_dataloader_iters = configuration["dataset_params"]["loader_params"][
        "max_iterations"
    ]
    global_step = configuration["training_params"]["global_step"]


    # writer = tensorboard.SummaryWriter(
    #     training_session["model_dir"].parent / "runs"
    # )

    stop_counter = 0
    model.train()
    for epoch in range(current_epoch, stop_epoch):

        total_loss = 0
        start = 0
        
        with ProgressBar(train_dataloader, max_dataloader_iters) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{stop_epoch}]")
            for index, (mixture, target) in enumerate(pbar):
                loading_time = time.time() - start
                start = time.time()

                mixture, target = mixture.to(device), target.to(device)
                mixture = model.process_input(mixture)
                target = model.process_input(target)
                stft_time = time.time() - start
                mask = model.forward(mixture)
                model.backward(mask, mixture, target)
                model.optimizer_step()

        #         # writer.add_scalar("Loss/train", model.loss.item(), global_step)

        #         # iter_losses.append(loss.item())
                pbar.set_postfix(
                    {
                        "loss": model.loss.item(),
                        # "loading_time": f"{round(loading_time, 2)}s",
                        "stft_time": f"{round(stft_time, 2)}s"
                    }
                )
                total_loss += model.loss.item()

                global_step += 1
                start = time.time()
            
                # break after seeing max_iter * batch_size samples
                if index >= max_dataloader_iters:
                    pbar.set_postfix(loss=total_loss / max_dataloader_iters)
                    pbar.clear()
                    break
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))


        # epoch_losses.append(total_loss / max_iters)

        # additional validation step for early stopping
        # val_loss, val_steps = cross_validate(
        #     model,
        #     val_dataloader,
        #     criterion,
        #     max_iters,
        #     writer,
        #     num_fft,
        #     window_size,
        #     hop_length,
        #     val_steps,
        #     device,
        # )
        # val_losses.append(val_loss)

        # update current training environment/model state
        # training_session["current_epoch"] = epoch
        # training_session["global_steps"] = global_steps
        # training_session["state_dict"] = model.state_dict()
        # training_session["optimizer"] = optimizer.state_dict()
        # training_session["iter_losses"] = iter_losses
        # training_session["epoch_losses"] = epoch_losses
        # training_session["val_losses"] = val_losses
        # training_session["trained"] = True

        # take snapshot and save to checkpoint directory
        # checkpoint_handler(training_session,
        #                    training_session['model_dir'] / 'checkpoints',
        #                    display=(epoch - 1) % 10 == 0)

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
