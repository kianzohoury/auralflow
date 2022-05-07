from visualizer.progress import ProgressBar
from torch.utils.data import DataLoader
from models.base import SeparationModel
from torch.utils.tensorboard import SummaryWriter
from torch import autocast


def cross_validate(
    model,
    writer,
    val_dataloader: DataLoader,
    max_iters: int,
    epoch: int,
    stop_epoch: int,
):
    """Performs cross validation."""

    max_iters = len(val_dataloader.dataset) // val_dataloader.batch_size
    # max_iters = 10

    model.eval()
    with ProgressBar(val_dataloader, max_iters) as pbar:
        pbar.set_description(f"Epoch [{epoch}/{stop_epoch}] val")
        total_loss = 0
        for index, (mixture, target) in enumerate(pbar):

            model.set_data(mixture, target)
            model.test()

            model.backward()
            batch_loss = model.get_batch_loss()

            pbar.set_postfix({"loss": batch_loss})
            total_loss += batch_loss

            writer.add_scalars(
                "Loss/val", {"l1_kl": batch_loss}, epoch * max_iters + index
            )

            if index == max_iters:
                pbar.set_postfix({"avg_loss": total_loss / max_iters})
                pbar.clear()
                break

    model.val_losses.append(total_loss / max_iters)
    pbar.set_postfix({"avg_loss": total_loss / max_iters})
