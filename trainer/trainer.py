import torch
from torch.utils.data import DataLoader

from utils.progress_bar import ProgressBar


def run_training_loop():
    pass













def cross_validate(model: torch.nn.Module, data_loader: DataLoader,
                   criterion: torch.nn.Module, max_iters: int,
                   device: str = 'cpu') -> float:
    r"""Cross validates a model's performance.

    Designed to be called after each training epoch to prevent over-fitting
    on the training set, and signal early stopping.

    Returns:
        (float): The batch-wise average validation loss.
    """
    num_iterations = max_iters
    total_loss = 0

    model.eval()
    with ProgressBar(data_loader, num_iterations, train=False) as pbar:
        for index, (mixture, target) in enumerate(pbar):
            mixture, target = mixture.to(device), target.to(device)

            mixture_stft = torch.stft(mixture.squeeze(1).squeeze(-1), 1023, 518, 1023, onesided=True, return_complex=True)
            target_stft = torch.stft(target.squeeze(1).squeeze(-1), 1023, 518, 1023, onesided=True, return_complex=True)

            # reshape data
            mixture_mag, target_mag = torch.abs(mixture_stft), torch.abs(target_stft)
            mixture_phase = torch.angle(mixture_stft)

            mixture_mag = mixture_mag.unsqueeze(-1)
            target_mag = target_mag.unsqueeze(-1)

            with torch.no_grad():

                # generate soft mask
                mask = model(mixture_mag)['mask']

                estimate = mask * mixture_mag

                # estimate source(s) and record loss
                loss = criterion(estimate, target_mag)

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            if index >= num_iterations:
                pbar.set_postfix(loss=round(total_loss / num_iterations, 3))
                pbar.clear()
                break

    return total_loss / num_iterations
