"""
trainer.py

This module contains a class to help with model training for AI/ML projects

Course: CSCI 357 - AI and Neural Networks
Author: Alex Searle
Date: 02/17/2026

"""
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from typing import Callable, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import os
import torchmetrics

from my_engine.config import TrainerConfig, ModelConfig, MetricsConfig
from my_engine.utils import accuracy_from_logits, make_lr_scheduler, get_preds

METRIC_REGISTRY = {
    'accuracy': torchmetrics.Accuracy,
    'f1': torchmetrics.F1Score,
    'precision': torchmetrics.Precision,
    'recall': torchmetrics.Recall,
    'mae': torchmetrics.MeanAbsoluteError,
    'mse': torchmetrics.MeanSquaredError,
    'r2': torchmetrics.R2Score,
}


def _build_metric_collections(config: MetricsConfig, device: torch.device) -> torchmetrics.MetricCollection:
    if config.task not in ['binary', 'multiclass', 'regression']:
        raise ValueError(f"Task {config.task} is not supported")

    metric_list = []
    if config.task == 'binary':
       for metric in config.names:
           metric_list.append(METRIC_REGISTRY[metric](
               task=config.task
           ).to(device))
    elif config.task == 'multiclass':
        for metric in config.names:
            metric_list.append(METRIC_REGISTRY[metric](
                task=config.task,
                num_classes=config.num_classes,
                average=config.average,
            ).to(device))
    else:
        for metric in config.names:
            metric_list.append(METRIC_REGISTRY[metric]().to(device))

    return torchmetrics.MetricCollection(metric_list)


class Trainer:
    """
    This is a class to encapsulate the training and evaluation of a Neural Network.

    Parameters:
        model: The model to be trained
        optimizer: The optimizer to be used for training
        criterion: The criterion to be used for calculating the loss
        config: TrainerConfig
        run: WandB Run for logging
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig = TrainerConfig(),
        run: Optional[wandb.Run] = None,
        metrics_config: Optional[MetricsConfig] = None,
    ) -> None:
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.run = run
        self.metric_config = metrics_config

        # Initialize learning rate scheduler if enabled
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = make_lr_scheduler(self.optimizer, self.config)

        # Initialize state variables for checkpointing and early stopping
        self.start_epoch = 0                # Use for resuming training from checkpoint
        self.current_epoch = 0              # Tracks current epoch during training
        self.best_val_loss = float('inf')   # Default best validation loss
        self.patience_counter = 0           # How many epochs without improvement before stopping

        if self.metric_config:
            self.train_metrics = _build_metric_collections(self.metric_config, self.config.device)
            self.val_metrics = _build_metric_collections(self.metric_config, self.config.device)

        if self.run:
            wandb.watch(self.model)

    def train_one_epoch(self, train_loader) -> Tuple[float, float, dict|None]:
        """
        Train the model for a single epoch
        :param train_loader: the training data to use
        :return: average loss
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        if self.config.metrics:
            metric_dict = {f"train_{x}": 0 for x in self.config.metrics.keys()}
            for _, metric in self.config.metrics.items():
                metric.to(self.config.device)
        else:
            metric_dict = None

        if self.metric_config:
            self.train_metrics.reset()

        # Iterate through the batches in the training loader
        for inputs, targets in train_loader:
            batch_acc = 0
            # Move the data to the correct device
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)

            # Train the model by doing a forward pass then back propagating the loss
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if self.metric_config and self.metric_config.task == 'regression':
                loss = self.criterion(outputs, targets)
            elif outputs.shape[1] == 1:
                loss = self.criterion(outputs.squeeze(), targets)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            if self.config.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_value)
            self.optimizer.step()

            # Keep track of the loss and number of samples seen
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Keep track of the accuracy
            if self.metric_config and self.metric_config.task == 'regression':
                pass
            else:
                acc = accuracy_from_logits(outputs, targets)
                total_acc += acc * batch_size

            # Calculate torchmetrics
            if self.config.metrics:
                preds = get_preds(outputs)
                for _, metric in self.config.metrics.items():
                    metric(preds, targets)

            if self.metric_config:
                if self.metric_config.task != 'regression':
                    preds = get_preds(outputs)
                    self.train_metrics.update(preds, targets)
                else:
                    self.train_metrics.update(outputs, targets)

        if self.config.metrics:
            for key, metric in self.config.metrics.items():
                metric_dict[f"train_{key}"] = metric.compute().item()
                metric.reset()

        if self.metric_config:
            output = self.train_metrics.compute()
            metric_dict = output

        if total_samples == 0:
            return 0.0, 0.0, None

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc, metric_dict

    def validate(self, val_loader) -> Tuple[float, float, dict|None]:
        """
        Validate the model on test data
        :param val_loader: data to be used for model validation
        :return: average loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        with torch.no_grad():
            # Iterate through the batches
            if self.config.metrics:
                metric_dict = {f"val_{x}": 0 for x in self.config.metrics.keys()}
                for _, metric in self.config.metrics.items():
                    metric.to(self.config.device)
            else:
                metric_dict = None

            if self.metric_config:
                self.val_metrics.reset()

            for X_batch, y_batch in val_loader:
                batch_acc = 0
                # Move the tensors to the correct device
                X_batch = X_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)

                # Do a forward pass and calculate the accuracy
                logits = self.model(X_batch)
                if self.metric_config and self.metric_config.task == 'regression':
                    loss = self.criterion(logits, y_batch)
                elif logits.shape[1] == 1:
                    loss = self.criterion(logits.squeeze(), y_batch)
                else:
                    loss = self.criterion(logits, y_batch)

                if self.metric_config and self.metric_config.task == 'regression':
                    pass
                else:
                    batch_acc = accuracy_from_logits(logits, y_batch)

                # Keep track of the overall accuracy and number of instances seen
                batch_size = X_batch.size(0)
                running_loss += loss.item() * batch_size
                running_acc += batch_acc * batch_size
                total_samples += batch_size
                # Calculate torchmetrics
                if self.config.metrics:
                    preds = get_preds(logits)
                    for _, metric in self.config.metrics.items():
                        metric(preds, y_batch)
                if self.metric_config:
                    if self.metric_config.task != 'regression':
                        preds = get_preds(logits)
                        self.val_metrics.update(preds, y_batch)
                    else:
                        self.val_metrics.update(logits, y_batch)

            if self.config.metrics:
                for key, metric in self.config.metrics.items():
                    metric_dict[f"val_{key}"] = metric.compute().item()
                    metric.reset()
            if self.metric_config:
                output = self.val_metrics.compute()
                metric_dict = output

        if total_samples == 0:
            return 0.0, 0.0, None

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples
        return avg_loss, avg_acc, metric_dict

    def fit(self, train_loader, val_loader, resume_from_last_checkpoint: bool = False, override_num_epochs: int = None) -> Dict[str, float]:
        """
        Fit and train the model and the given data
        :param train_loader: the data to be used for training
        :param val_loader: the data to be used for validation
        :return: the info of the training session
        """
        # Resume from checkpoint if specified
        if resume_from_last_checkpoint:
            self.load_checkpoint()

        if override_num_epochs:
            self.config.num_epochs = override_num_epochs

        if self.run:
            params = self.model.num_parameters()
            self.run.config.update(asdict(self.config))
            self.run.config.update(asdict(self.model.config))
            self.run.config.update({
                "num_parameters": params[0],
                "num_train_parameters": params[1]
            })
        # Sanity check: Verify the batch sizes match the config supplied
        if hasattr(train_loader, 'batch_size') and train_loader.batch_size is not None:
            if train_loader.batch_size != self.config.trainer_batch_size:
                raise ValueError(f"Train loader batch size ({train_loader.batch_size}) does not match config ({self.config.trainer_batch_size})")

        val_losses = []
        train_losses = []
        val_accs = []
        train_accs = []
        # Train the model for the number of epochs in the config
        for self.current_epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss, train_acc, train_metric_dict = self.train_one_epoch(train_loader)
            val_loss, val_acc, test_metric_dict = self.validate(val_loader)

            val_losses.append(val_loss)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(
                f"Epoch {self.current_epoch}: Train Loss={train_loss:.4f},"
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc * 100:.2f}%,"
                f"Metrics: Train: {train_metric_dict}, Test: {test_metric_dict}"
            )
            if self.config.metrics:
                print(f"val_metrics: {test_metric_dict}")
            if self.run:
                self.run.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                    "epoch": self.current_epoch+1,
                })
                if self.config.metrics:
                    self.run.log(train_metric_dict)
                    self.run.log(test_metric_dict)
                if self.metric_config:
                    train_update = {}
                    for key, metric in train_metric_dict.items():
                        train_update[f"train_{key}"] = metric
                    self.run.log(train_update)
                    val_update = {}
                    for key, metric in test_metric_dict.items():
                        val_update[f"val_{key}"] = metric
                    self.run.log(val_update)
            # Save a checkpoint every n epochs
            if (self.current_epoch) + 1 % self.config.checkpoint_save_interval:
                self.save_checkpoint()

            if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                self.save_checkpoint(True)
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter == self.config.early_stopping_patience:
                break

            # Step the learning rate scheduler
            if self.scheduler is not None:
                if self.config.scheduler_type == "reduce_on_plateau":
                    # ReduceLROnPlateau needs the validation loss
                    self.scheduler.step(val_loss)
                else:
                    # Other schedulers just need to know an epoch completed
                    self.scheduler.step()

                # Log current learning rate to W&B
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.run is not None:
                    self.run.log({"learning_rate": current_lr}, step=self.current_epoch + 1)

        # Return the final results of training
        info_dict = { "num_epochs": self.current_epoch+1,
                 "train_loss": train_loss,
                "train_acc": train_acc,
                 "val_loss": val_loss,
                 "val_acc": val_acc,
                    "val_losses": val_losses,
                      "train_losses": train_losses,
                      "train_accs": train_accs,
                      "val_accs": val_accs,
                    }
        if self.config.metrics:
            info_dict.update(train_metric_dict)
            info_dict.update(test_metric_dict)
        if self.metric_config:
            info_dict.update(train_metric_dict)
            info_dict.update(test_metric_dict)
        return info_dict

    def finish_run(self) -> None:
        if self.run:
            wandb.unwatch(self.model)
            self.run.finish()

    def save_checkpoint(self, is_best: bool = False) -> None:
        """
            Save a training checkpoint to disk.

            This method serializes the current training state into a dictionary and writes it
            using `torch.save`. The checkpoint includes:
              - Model parameters (`model_state_dict`) needed to restore weights.
              - Optional model architecture/configuration (`model_architecture`) if the model
                implements `get_architecture_config()`. Otherwise this field is `None`.
              - Trainer configuration (`trainer_config`) stored as a plain dictionary.
              - Optimizer state (`optimizer_state_dict`) needed to resume training with the
                same momentum/Adam moments, etc.
              - Early-stopping/training bookkeeping (`best_val_loss`, `epoch`, `patience_counter`).

            Two files may be written:
              - A "last" checkpoint saved every time this method is called.
              - If `is_best=True`, an additional "best" checkpoint saved to a separate filename.

            Parameters
            ----------
            is_best : bool, default=False
                If True, also save the checkpoint as the current best checkpoint (e.g., when
                validation loss improves), in addition to saving the "last" checkpoint.

            Returns
            -------
            None
                This function writes checkpoint files to disk and does not return anything.
            """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_last_filename)

        checkpoint = {
            # Model weights (the numbers)
            'model_state_dict': self.model.state_dict(),

            # Architecture specification (the blueprint)
            'model_architecture': self.model.get_architecture_config() if hasattr(self.model,
                                                                                    'get_architecture_config') else None,

            # Training state
            'trainer_config': asdict(self.config),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.current_epoch,
            'patience_counter': self.patience_counter,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_best_filename)
            torch.save(checkpoint, best_path)
            print(f"--> New best checkpoint saved: {best_path}")
            print(f"--> Also saving as last checkpoint: {filepath}")
        else:
            print(f"--> Saving checkpoint: {filepath}")
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, retrieve_best: bool = False):
        if retrieve_best:
            filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_best_filename)
        else:
            filepath = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_last_filename)

        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.start_epoch = checkpoint['epoch']
        self.patience_counter = checkpoint['patience_counter']
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_run()