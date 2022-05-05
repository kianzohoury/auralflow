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
                "Loss/val",
                {"batch_64_lr_0005_VAE_1024": batch_loss},
                # global_step,
            )
            pbar.set_postfix({"avg_loss": batch_loss})
            total_loss += batch_loss

            # global_step += 1

            if index >= max_iters:
                pbar.set_postfix({"avg_loss": total_loss / max_iters})
                pbar.clear()
                break

    pbar.set_postfix({"avg_loss": total_loss / max_iters})
