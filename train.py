from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter

from datasets import create_audio_dataset, load_dataset
from models import create_model
from utils import load_config
from validate import cross_validate
from visualizer.progress import ProgressBar


def main(config_filepath: str):
    """Runs training script given a configuration file."""

    # Load configuration file.
    print("-" * 79 + "\nReading configuration file...")
    configuration = load_config(config_filepath)
    training_params = configuration["training_params"]
    dataset_params = configuration["dataset_params"]
    visualizer_params = configuration["visualizer_params"]
    print("Successful.")

    # Load training/validation data.
    print("-" * 79 + "\nLoading training data...")
    train_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="train",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=dataset_params["max_num_samples"],
        max_num_tracks=dataset_params["max_num_tracks"]
    )
    val_dataset = create_audio_dataset(
        dataset_params["dataset_path"],
        split="val",
        targets=dataset_params["targets"],
        chunk_size=dataset_params["sample_length"],
        num_chunks=dataset_params["max_num_samples"],
        max_num_tracks=dataset_params["max_num_tracks"]
    )
    train_dataloader = load_dataset(train_dataset, training_params)
    val_dataloader = load_dataset(val_dataset, training_params)
    print(
        f"Successful. Loaded {len(train_dataset)} training and "
        f"{len(val_dataset)} validation samples of length "
        f"{dataset_params['sample_length']}s."
    )

    # Load source separation model.
    print("-" * 79 + "\nLoading model...")
    model = create_model(configuration)
    # model.setup()
    print("Done.")

    writer = SummaryWriter(log_dir=visualizer_params["logs_path"])
    start_epoch = training_params["last_epoch"] + 1
    stop_epoch = start_epoch + training_params["max_epochs"]
    global_step = configuration["training_params"]["global_step"]
    max_iters = len(train_dataloader)
    print("Starting training...\n" + "-" * 79)

    for epoch in range(start_epoch, stop_epoch):
        print(f"Epoch {epoch}/{stop_epoch}", flush=True)
        total_loss = 0
        train_loss = []
        model.train()
        with ProgressBar(train_dataloader, total=max_iters) as pbar:
            for idx, (mixture, target) in enumerate(pbar):
                # Cast precision if necessary to increase training speed.
                # with autocast(device_type=model.device):

                # Process data, run forward pass.
                model.set_data(mixture, target)
                model.forward()

                # Calculate mini-batch loss and run backprop.
                batch_loss = model.compute_loss()
                total_loss += batch_loss
                train_loss.append(batch_loss)
                model.backward()

                model.mid_epoch_callback(writer=writer, global_step=global_step)

                # nn.utils.clip_grad_norm_(
                #     model.model.parameters(), max_norm=1.0
                # )

                # Update model parameters.
                model.optimizer_step()
                global_step += 1
                
                # Display and log the loss.
                pbar.set_postfix({"loss": round(batch_loss, 6)})
                writer.add_scalar(
                    "Loss/train/iter_avg", batch_loss, global_step
                )

        avg_loss = sum(train_loss) / len(train_loss)
        pbar.set_postfix({"loss": round(avg_loss, 6)})
        # Store epoch-average loss.
        model.train_losses.append(avg_loss)
        writer.add_scalar(
            "Loss/train/epoch_avg", avg_loss, epoch,
        )

        # Validate updated model.
        cross_validate(
            model=model,
            val_dataloader=val_dataloader,
        )

        print("avg train loss:", model.train_losses[-1])
        print("avg val loss:", model.val_losses[-1])
        print("sdr:")
        print("snr:")
        print("-" * 79)

        # Decrease lr if necessary.
        stop_early = model.scheduler_step()

        # Log validation loss.
        writer.add_scalar(
            "Loss/val/epoch_avg", model.val_losses[-1], epoch
        )

        writer.add_scalars(
            "Loss",
            {"train": model.train_losses[-1], "val": model.val_losses[-1]},
            epoch
        )

        if stop_early:
            print("No improvement. Stopping training early...")
            break

        # Only save the best model.
        if model.is_best_model:
            model.save_model(global_step=epoch)
            model.save_optim(global_step=epoch)

        mixture, target = next(iter(val_dataloader))
        model.post_epoch_callback(
            mixture, target, writer, epoch
        )

    writer.close()
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Model training script.")
    parser.add_argument(
        "config_filepath", type=str, help="Path to a configuration file."
    )
    args = parser.parse_args()
    main(args.config_filepath)
