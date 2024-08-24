"""Runs the training process with the specified parameters."""
import os
from typing import Optional
import time  # Import time module for timing

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from Numbersense.config import Dataloader_Parameters, Experiment_Parameters
from Numbersense.data_loaders.training_dataloader import TrainDataloader
from Numbersense.utilities.helpers import _get_wandb_token, get_compute, getenv
from Numbersense.utilities.navigator import Navigator

if getenv("WANDB", False):
    import wandb


class Runner:
    def __init__(
        self,
        dataset_path: str,
        objective_function: str,
        background_mode: str,
        experiment_parameters: Experiment_Parameters,
        dataloader_parameters: Dataloader_Parameters,
        model: nn.Module = None,
        model_num: Optional[int] = None,
    ):
        assert model is not None

        self.loss_history = []
        self.objective_function = objective_function

        # Select hardware to run computations on
        self.device = get_compute()
        self.model = model.to(self.device)

        self.data_dir = dataset_path

        # Generate model save directory
        model_dir = Navigator.get_model_path(
            experiment_parameters.save_directory,
            objective_function,
            background_mode,
            self.model.name,
        )
        os.makedirs(model_dir, exist_ok=True)
        save_dir = (
            os.path.join(model_dir, f"set_{str(model_num)}")
            if model_num is not None
            else Navigator.get_next_set(model_dir)
        )
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        print("\nCreating dataloader...")
        self.loader_obj = TrainDataloader(
            dataloader_parameters,
            data_dir=self.data_dir,
            save_dir=self.save_dir,
            classf=False,
            save=False,
            objective_function=objective_function,
        )

    def train_embedding_model_unsupervised_epoch(
        self,
        optimizer: any,
        loss_fn: any,
        lr_scheduler: any,
        epoch: int,
    ):
        # Store plain, untrained model
        if epoch == 0 and getenv("SAVE_UNTRAINED", 0):
            checkpoint = {
                "epoch": 0,
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": lr_scheduler,
                "loss_history": [],
            }
            torch.save(checkpoint, os.path.join(self.save_dir, "untrained.pt"))

        self.model.train()

        self.loader_obj.load_data(epoch=(epoch % 30))
        train_data_loader = torch.utils.data.DataLoader(
            self.loader_obj,
            batch_size=self.loader_obj.batch_size,
            shuffle=False,
            num_workers=getenv("NUM_WORKERS", os.cpu_count()),
        )

        if getenv("MIXED_PRECISION_TRAINING", 0):
            scaler = GradScaler()

        epoch_start = time.time()
        for batch_idx, (data, targets) in enumerate(train_data_loader):
            for k in range(len(data)):
                # Drop alpha channel before moving to GPU
                data[k] = data[k].to(self.device)
            targets = targets.type(torch.LongTensor).to(self.device)
            optimizer.zero_grad()

            self.model.to(self.device)

            if getenv('MIXED_PRECISION_TRAINING', 0):
                with autocast():
                    output_same = self.model(data[0][:, :3, :, :], data[1][:, :3, :, :])

                    target_same = torch.squeeze(targets[:, 0])
                    loss = loss_fn(output_same, torch.argmax(target_same, dim=1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output_same = self.model(data[0][:, :3, :, :], data[1][:, :3, :, :])
                target_same = torch.squeeze(targets[:, 0])
                loss = loss_fn(output_same, torch.argmax(target_same, dim=1))

                loss.backward()
                optimizer.step()

            if getenv("WANDB", 0): wandb.log({"loss": loss})
            if batch_idx % 10 == 0: self.loss_history.append(loss.cpu().detach().numpy())

        epoch_end = time.time()
        if getenv("VERBOSE", 0): print(f"Epoch {epoch:02d} training time: {(epoch_end - epoch_start):.2f}s, Loss: {loss.item():.2f}")

    def train_embedding_model_unsupervised(
        self,
        optimizer: any = None,
        loss_fn: any = None,
        scheduler: any = None,
        num_epochs: int = 30,
        plot_loss: bool = True,
    ):
        assert num_epochs > -1

        # Save the trained model
        final_path = os.path.join(self.save_dir, "final.pt")

        if not os.path.exists(final_path):
            if getenv("WANDB", 0):
                token, model_id = _get_wandb_token()
                wandb.login(key=token)
                wandb.init(
                    project=self.objective_function,
                    name=model_id,
                    config={
                        "learning_rate": 0.001,
                        "architecture": "SiameseClassificationNetwork",
                        "dataset": self.objective_function,
                        "epochs": num_epochs,
                    },
                )

            # Outer training loop
            for epoch in range(num_epochs):
                self.train_embedding_model_unsupervised_epoch(
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    lr_scheduler=scheduler,
                    epoch=epoch,
                )

                # If there is a scheduler, start it.
                if scheduler is not None:
                    scheduler.step()

            # Save the trained model
            final_path = os.path.join(self.save_dir, "final.pt")

            final = {
                "epoch": "final",
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": scheduler,
                "loss_history": self.loss_history,
            }
            torch.save(final, final_path)

            if getenv("WANDB", 0):
                wandb.finish()

            if plot_loss:
                plt.plot(self.loss_history)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title(f"Training loss for {num_epochs} Epochs")
                figure_path = os.path.join(self.save_dir, "figures")
                filename = f"loss_{self.model.unique_id}"
                os.makedirs(figure_path, exist_ok=True)
                plt.savefig(os.path.join(figure_path, filename))
        else:
            print("\nThis model has already been trained on the requested dataset!")
