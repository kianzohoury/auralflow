from pickle import TRUE
import sys
import time

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



#
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from utils.progress_bar import ProgressBar
#
# #
# def run_training_loop():
#     pass

#
# def cross_validate(model: torch.nn.Module, data_loader: DataLoader,
#                    criterion: torch.nn.Module, max_iters: int,
#                    writer: SummaryWriter, num_fft: int, window_size: int,
#                    hop_length: int, val_steps: int, device: str = 'cpu') -> float:
#     r"""Cross validates a model's performance.
#
#     Designed to be called after each training epoch to prevent over-fitting
#     on the training set, and signal early stopping.
#
#     Returns:
#         (float): The batch-wise average validation loss.
#     """
#     num_iterations = max_iters
#     total_loss = 0
#
#     model.eval()
#     with ProgressBar(data_loader, num_iterations, train=False) as pbar:
#         for index, (mixture, target) in enumerate(pbar):
#             mixture, target = mixture.to(device), target.to(device)
#
#             mixture_stft = torch.stft(mixture.squeeze(1).squeeze(-1),
#                                       num_fft - 1, hop_length, window_size - 1,
#                                       onesided=True, return_complex=True)
#             target_stft = torch.stft(target.squeeze(1).squeeze(-1), num_fft - 1,
#                                      hop_length, window_size - 1, onesided=True,
#                                      return_complex=True)
#
#             # reshape data
#             mixture_mag, target_mag = torch.abs(mixture_stft), torch.abs(target_stft)
#             mixture_phase = torch.angle(mixture_stft)
#
#             mixture_mag = mixture_mag.unsqueeze(-1)
#             target_mag = target_mag.unsqueeze(-1)
#
#             with torch.no_grad():
#
#                 # generate soft mask
#                 mask = model(mixture_mag)['mask']
#
#                 estimate = mask * mixture_mag
#
#                 # estimate source(s) and record loss
#                 loss = criterion(estimate, target_mag)
#
#             total_loss += loss.item()
#             writer.add_scalar("Val/Loss", loss.item(), val_steps)
#             pbar.set_postfix(loss=loss.item())
#
#             val_steps += 1
#
#             if index >= num_iterations:
#                 pbar.set_postfix(loss=round(total_loss / num_iterations, 3))
#                 pbar.clear()
#                 break
#
#     return (total_loss / num_iterations, val_steps)