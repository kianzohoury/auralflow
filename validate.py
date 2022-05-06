from visualizer.progress import ProgressBar
from torch.utils.data import DataLoader
from models.base import SeparationModel
from torch.utils.tensorboard import SummaryWriter


def cross_validate(
    model,
    val_dataloader: DataLoader,
    max_iters: int,
    writer: SummaryWriter,
):
    """Performs cross validation."""

    global_val_step = len(model.val_losses) * max_iters

    model.eval()
    with ProgressBar(val_dataloader, max_iters) as pbar:
        pbar.set_description("Evaluating...")
        total_loss = 0
        for index, (mixture, target) in enumerate(pbar):

            model.set_data(mixture, target)
            model.forward()
            model.backward()

            batch_loss = model.get_batch_loss()

            writer.add_scalars(
                "Loss/val", {"l1_kl": batch_loss}, global_val_step
            )

            pbar.set_postfix({"avg_loss": batch_loss})
            total_loss += batch_loss
            global_val_step += 1

            if index >= max_iters:
                pbar.set_postfix({"avg_loss": total_loss / max_iters})
                pbar.clear()
                break

    model.val_losses.append(total_loss / max_iters)
    pbar.set_postfix({"avg_loss": total_loss / max_iters})
