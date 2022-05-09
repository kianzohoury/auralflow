from torch import autocast
from torch.utils.data import DataLoader
from visualizer.progress import ProgressBar


def cross_validate(model, val_dataloader: DataLoader) -> None:
    """Validates network updates on the validation set."""

    num_iters = len(val_dataloader)

    model.eval()
    with ProgressBar(val_dataloader, total=num_iters, desc="val:") as pbar:
        total_loss = 0
        for idx, (mixture, target) in enumerate(pbar):
            # # Cast precision if necessary to increase training speed.
            # with autocast(device_type=model.device):
            model.set_data(mixture, target)
            model.test()

            # Compute batch-wise loss.
            model.backward()
            total_loss += model.get_batch_loss()
            # Display loss.
            pbar.set_postfix({"loss": model.get_batch_loss()})

            if idx == num_iters:
                pbar.clear()
                break

    # Store epoch-average validation loss.
    model.val_losses.append(total_loss / num_iters)
