


def cross_validate(model, val_dataloader):
    with ProgressBar(train_dataloader, iters) as pbar:
        pbar.set_description(f"Epoch [{epoch}/{stop_epoch}]")
        for index, (mixture, target) in enumerate(pbar):

            model.set_data(mixture, target)
            model.forward()
            model.backward()
            model.optimizer_step()

            batch_loss = model.get_batch_loss()

            writer.add_scalars(
                "Loss/train",
                {"batch_64_lr_0005_VAE_1024": batch_loss},
                global_step,
            )
            pbar.set_postfix({"avg_loss": batch_loss})
            total_loss += batch_loss

            global_step += 1

            # break after seeing max_iter * batch_size samples
            if index >= iters:
                pbar.set_postfix({"avg_loss": total_loss / iters})
                pbar.clear()
                break

    pbar.set_postfix({"avg_loss": total_loss / iters})