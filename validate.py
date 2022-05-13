from torch import autocast
from torch.utils.data import DataLoader
from visualizer.progress import ProgressBar
import torch


def cross_validate(model, val_dataloader: DataLoader) -> None:
    """Validates network updates on the validation set."""

    num_iters = len(val_dataloader)
    val_loss = []

    model.eval()
    with ProgressBar(val_dataloader, total=num_iters) as pbar:
        total_loss = 0
        for idx, (mixture, target) in enumerate(pbar):
            # # Cast precision if necessary to increase training speed.
            # with autocast(device_type=model.device):
            model.set_data(mixture, target)
            with torch.no_grad():
                model.test()
                # Compute batch-wise loss.
                batch_loss = model.compute_loss()
                total_loss += batch_loss
                val_loss.append(batch_loss)

            # Display loss.
            pbar.set_postfix({"loss": batch_loss})

    # Store epoch-average validation loss.
    # model.val_losses.append(total_loss / num_iters)
    model.val_losses.append(sum(val_loss) / len(val_loss))
