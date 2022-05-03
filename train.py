import time
from argparse import ArgumentParser

import torch
import numpy as np

from datasets import create_dataset, load_dataset
from models import create_model
from utils import load_config
from utils.progress_bar import ProgressBar
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def main(config_filepath: str):

    print("Reading configuration file...")
    configuration = load_config(config_filepath)

    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    loader_params = dataset_params["loader_params"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    train_dataset = create_dataset(
        dataset_params=dataset_params, subset="train"
    )
    # val_dataset = train_dataset.split(val_split=dataset_params["val_split"])
    # train_dataloader = load_dataset(
    #     dataset=train_dataset, loader_params=dataset_params["loader_params"]
    # )
    train_dataloader = DataLoader(
        train_dataset, num_workers=10, pin_memory=True,
        persistent_workers=True, batch_size=128, prefetch_factor=4)
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
    max_iters_per_epoch = loader_params["max_iterations"]
    global_step = configuration["training_params"]["global_step"]

    writer = SummaryWriter("runs_vocals_1")

    model.train()
    for epoch in range(current_epoch, stop_epoch):
        total_loss = 0
        with ProgressBar(train_dataloader, max_iters_per_epoch) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{stop_epoch}]")
            for index, (mixture, target) in enumerate(pbar):

                mixture, target = mixture.to(device), target.to(device)

                model.set_data(mixture, target)
                model.forward()
                model.backward()
                model.optimizer_step()

                writer.add_scalars("Loss/train", {"batch_128_60_lr_0005_VAE": model.losses[-1]}, global_step)
                pbar.set_postfix({"avg_loss": model.batch_losses[-1]})
                total_loss += model.losses[-1]

                global_step += 1
                # start = time.time()

                # break after seeing max_iter * batch_size samples
                if index >= max_iters_per_epoch:
                    pbar.set_postfix({"avg_loss": total_loss / max_iters_per_epoch})
                    pbar.clear()
                    break
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
