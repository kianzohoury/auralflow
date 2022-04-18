import sys
import time

import torch
import torch.nn as nn
import numpy as np
import torchinfo

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

def main(config: dict):
    """Main training method.

    Args:
        config (dict): Dictionary containing the training parameters.
    """

    # Set seeds.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # ignore tensorboard for now

    train_dataset = AudioFolder(**config['dataset'], subset='train')
    val_dataset = train_dataset.split(val_split=config['training']['val_split'])

    if config['training']['cuda'] and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    # optimizer = config['training']['optimizer']
    num_workers = config['training']['num_workers']
    pin_mem = config['training']['pin_memory']
    persistent = config['training']['persistent_workers']
    patience = config['training']['patience']

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

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    global_steps = 0
    global_clock = 0
    iter_history = []
    epoch_history = []
    validation_history = []
    best_val_loss = float("inf")
    stop_counter = 0

    # initialize model state
    model_state = {
        'model_base' : model.__class__.__name__,
        'device' : device,
        'epoch' : 0,
        'global_steps' : 0,
        # 'max_iter' : max_iter,
        'avg_epoch_time' : float("inf"),
        'total_training_time' : float("inf"),
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'lr' : lr,
        'criterion' : criterion,
        'batch_size' : batch_size,
        'iter_history' : iter_history,
        'epoch_history' : epoch_history,
        'val_history' : validation_history,
        'final_train_loss' : float("inf"),
        'final_val_loss' : float("inf"),
        'best_val_loss' : float("inf"),
        'checkpoint_dir' : config['session']['checkpoint_folder']
    }

    num_iterations = config['training']['max_iters']

    print("=" * 90)
    print("Training session started...")
    print("=" * 90)

    model.train()
    for epoch in range(1, epochs + 1):

        total_loss = 0
        epoch_clock = 0

        with ProgressBar(train_dataloader, num_iterations) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{epochs}]")
            for index, (mixture, target) in enumerate(pbar):
                optimizer.zero_grad()

                mixture, target = mixture.to(device), target.to(device)

                mixture_stft = torch.stft(mixture.squeeze(1).squeeze(-1), 1023, 518, 1023, onesided=True, return_complex=True)
                target_stft = torch.stft(target.squeeze(1).squeeze(-1), 1023, 518, 1023, onesided=True, return_complex=True)

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
                iter_history.append(loss.item())
                pbar.set_postfix(loss=round(loss.item(), 3))

                # backpropagation/update step
                loss.backward()
                optimizer.step()

                global_steps += 1
                epoch_clock = pbar.format_dict['elapsed']

                # break after seeing max_iter * batch_size samples
                if index >= num_iterations:
                    pbar.set_postfix(loss=total_loss / num_iterations)
                    pbar.clear()
                    break

            global_clock += epoch_clock

        epoch_history.append(total_loss / num_iterations)

        # additional validation step for early stopping
        val_loss = cross_validate(model, val_dataloader, criterion,
                                  max_iters=num_iterations)
        validation_history.append(val_loss)

        # update current training environment/model state
        model_state['epoch'] = epoch
        model_state['global_steps'] = global_steps
        model_state['total_training_time'] = global_clock
        model_state['state_dict'] = model.state_dict()
        model_state['avg_epoch_time'] = global_clock / epoch
        model_state['optimizer'] = optimizer.state_dict()
        model_state['iter_history'] = iter_history
        model_state['epoch_history'] = epoch_history
        model_state['val_history'] = validation_history
        model_state['final_training_loss'] = total_loss / num_iterations
        model_state['final_val_loss'] = val_loss

        # take snapshot and save to checkpoint directory
        # checkpoint_handler(model_state, checkpoint_dir, display=(epoch - 1) % 10 == 0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state['best_val_loss'] = best_val_loss
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
        print(data_config_copy)
        for key in ['backend', 'audio_format']:
            data_config_copy.pop(key)
        data_config_copy['batch_size'] = config_dict['training']['batch_size']
        input_shape = transforms.get_data_shape(**data_config_copy)
        print(input_shape)
        torchinfo.summary(model, input_size=input_shape[:-1], depth=8)
    except Exception as e:
        print(e)
        raise e
        sys.exit(0)

    try:
        dataset = config.build.build_audio_folder(config_dict, args['dataset'])
    except FileNotFoundError as e:
        print(str(e))
    # print(dataset.__dict__)
    # print(vars(args))
    # unet = UNet()
    # torchinfo.summary(unet, input_size=(16, 512, 128, 1))

    # build_model(vars(args)['model'])

    # main(config)


