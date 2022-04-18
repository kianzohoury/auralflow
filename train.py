import sys
import time

import torch
import torch.nn as nn
import numpy as np
import torchinfo
from torch.utils import tensorboard
from utils.utils import checkpoint_handler

from pathlib import Path

import config.utils
from models.base_unet import UNet
from trainer.trainer import cross_validate
from utils.progress_bar import ProgressBar
from torch.utils.data.dataloader import DataLoader
from AudioFolder.datasets import AudioFolder
from argparse import ArgumentParser
import transforms
from yaml import YAMLError
from typing import List
import ruamel.yaml
import config.build
import yaml

from pprint import pprint

config_dir = Path(__file__).parent / 'config'
session_logs_file = config_dir / 'session_logs.yaml'

config_parser = ruamel.yaml.YAML(typ='safe', pure=True)
yaml_parser = ruamel.yaml.YAML()


def main(training_session: dict):
    """Main training method.

    Args:
        training_session (dict): Dictionary containing the training parameters.
    """

    # Set seeds.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if training_session['parameters']['cuda'] and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = training_session['model'].to(device)

    epochs = training_session['parameters']['epochs']
    batch_size = training_session['parameters']['batch_size']
    lr = training_session['parameters']['lr']
    optimizer = training_session['optimizer']
    num_workers = training_session['parameters']['num_workers']
    pin_mem = training_session['parameters']['pin_memory']
    persistent = training_session['parameters']['persistent_workers']
    patience = training_session['parameters']['patience']
    val_split = training_session['parameters']['val_split']
    max_iters = training_session['parameters']['max_iters']
    current_epoch = training_session['current_epoch']
    global_steps = training_session['global_steps']

    iter_losses = training_session['iter_losses']
    epoch_losses = training_session['epoch_losses']
    val_losses = training_session['val_losses']
    best_val_loss = training_session['best_val_loss']

    num_fft = training_session['data']['num_fft']
    window_size = training_session['data']['window_size']
    hop_length = training_session['data']['hop_length']

    train_dataset = training_session['dataset']
    val_dataset = train_dataset.split(val_split)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent
    )

    print("=" * 95)
    print("Training session started...")
    print("=" * 95)

    writer = tensorboard.SummaryWriter(training_session['model_dir'])
    stop_counter = 0
    model.train()
    for epoch in range(current_epoch, current_epoch + epochs + 1):

        total_loss = 0

        with ProgressBar(train_dataloader, max_iters) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{epochs}]")
            for index, (mixture, target) in enumerate(pbar):
                optimizer.zero_grad()

                mixture, target = mixture.to(device), target.to(device)

                mixture_stft = torch.stft(mixture.squeeze(1).squeeze(-1),
                                          num_fft - 1, hop_length,
                                          window_size - 1, onesided=True,
                                          return_complex=True)
                target_stft = torch.stft(target.squeeze(1).squeeze(-1),
                                         num_fft - 1, hop_length,
                                         window_size - 1, onesided=True,
                                         return_complex=True)

                # reshape data
                mixture_mag, target_mag = torch.abs(mixture_stft), torch.abs(target_stft)
                mixture_phase = torch.angle(mixture_stft)

                mixture_mag = mixture_mag.unsqueeze(-1)
                target_mag = target_mag.unsqueeze(-1)

                # generate soft mask
                mask = model(mixture_mag)['mask']

                estimate = mask * mixture_mag

                # estimate source(s) and record loss
                loss = criterion(estimate, target_mag)
                total_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_steps)

                iter_losses.append(loss.item())
                pbar.set_postfix(loss=round(loss.item(), 3))

                # backpropagation/update step
                loss.backward()
                optimizer.step()

                global_steps += 1

                # break after seeing max_iter * batch_size samples
                if index >= max_iters:
                    pbar.set_postfix(loss=total_loss / max_iters)
                    pbar.clear()
                    break

        epoch_losses.append(total_loss / max_iters)

        # additional validation step for early stopping
        val_loss = cross_validate(model, val_dataloader, criterion,
                                  max_iters, writer, num_fft,
                                  window_size, hop_length, epoch, device)
        val_losses.append(val_loss)

        # update current training environment/model state
        training_session['current_epoch'] = epoch
        training_session['global_steps'] = global_steps
        training_session['state_dict'] = model.state_dict()
        training_session['optimizer'] = optimizer.state_dict()
        training_session['iter_losses'] = iter_losses
        training_session['epoch_losses'] = epoch_losses
        training_session['val_losses'] = val_losses
        training_session['trained'] = True

        # take snapshot and save to checkpoint directory
        checkpoint_handler(training_session,
                           training_session['model_dir'] / 'checkpoints',
                           display=(epoch - 1) % 10 == 0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_session['best_val_loss'] = best_val_loss
            stop_counter = 0
        elif stop_counter < patience:
            stop_counter += 1
            epochs_left = patience - stop_counter + 1
            if epoch < epochs:
                print("=" * 90)
                print(f"Early Stopping: {epochs_left} epochs left if no " \
                      "improvement is made.")
                print("=" * 90)
        else:
            break

    print("=" * 90 + "\nTraining finished.")


if __name__ == "__main__":

    parser = ArgumentParser(description="Training script.")

    parser.add_argument('model', type=str, help="Model name to train.")
    parser.add_argument('--resume', '-r', type=str, help="Resume training.",
                        metavar='')
    parser.add_argument('--dataset', '-d', type=str,
                        help="Path of the dataset to train on.",
                        metavar='', required=True
                        )

    args = vars(parser.parse_args())

    model_name = args['model']
    logs = config.utils.get_session_logs(session_logs_file)
    session_dir = Path(logs['current']['location'])
    model_dir = session_dir / model_name
    config_dict = config.build.get_all_config_contents(model_dir)
    try:
        model = config.build.build_model(config_dict)
        print("Success: PyTorch model was built. Visualizing model...")
        time.sleep(3)
        data_config_copy = dict(config_dict['data'])
        for key in ['backend', 'audio_format']:
            data_config_copy.pop(key)
        data_config_copy['batch_size'] = config_dict['training']['batch_size']
        input_shape = transforms.get_data_shape(**data_config_copy)
        torchinfo.summary(model, input_size=input_shape[:-1], depth=8)
    except Exception as e:
        print(e)
        raise e

    try:
        dataset = config.build.build_audio_folder(config_dict, args['dataset'])
    except FileNotFoundError as e:
        raise e

    checkpoints_dir = model_dir / Path('checkpoints')

    if not args['model']:
        print("Error: cannot train model.")
        sys.exit(0)
    elif args['model'] and not checkpoints_dir.is_dir():
        training_settings = config_dict['training']
        optimizer = torch.optim.Adam(model.parameters(), training_settings['lr'])
        criterion = nn.L1Loss()
        global_steps = 0
        iter_losses = []
        epoch_losses = []
        val_losses = []
        best_val_loss = float("inf")

        training_session = {
            'model': model,
            'dataset': dataset,
            'data': config_dict['data'],
            'parameters': training_settings,
            'session_dir': session_dir,
            'model_dir': model_dir,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_dict': optimizer.state_dict(),
            'global_steps': global_steps,
            'criterion': criterion,
            'iter_losses': iter_losses,
            'epoch_losses': epoch_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'current_epoch': 0,
            'trained': False
        }
        checkpoints_dir.mkdir()
        torch.save(training_session, checkpoints_dir / f"{model_name}_latest.pth")
    else:
        training_session = torch.load(checkpoints_dir / f"{model_name}_latest.pth")
    main(training_session)
