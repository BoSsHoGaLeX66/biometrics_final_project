
import os
from functools import partial

import torch
import wandb
from typing import Optional, Union
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from my_engine.config import TrainerConfig, ModelConfig, ConvBlockConfig, ResidualBlockConfig, MetricsConfig
from my_engine.utils import build_model, make_optimizer
from my_engine.trainer import Trainer
from my_engine.data import get_dataloaders
from my_engine.text import text_collate_fn


def print_sweep_info(sweep_id: str) -> None:
    """
    This method takes in a sweep_id and retrieves it from the wandb api and prints out its current state
    :param sweep_id: the id of the weep we want
    :return: None
    """
    # Get the sweep from the API
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    # Print out the sweep info
    print(f"Sweep {sweep_id} has {len(sweep.runs)} runs")
    print(f"Sweep {sweep_id} expected {sweep.expected_run_count} runs")
    print(f"Sweep {sweep_id} current state is: {sweep.state}")

def terminate_sweep(sweep_id: str) -> bool:
    """
    Given a sweep_id checks if it is finished. If it is not finished it attempts to stop the sweep
    :param sweep_id:
    :return:
    """
    # Get the sweep from the API
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    print(f"Sweep {sweep_id} current state is: {sweep.state}")
    # If the sweep is not finished attempt to stop it
    if sweep.state != "FINISHED":
        s = sweep.entity + '/' + sweep.project + '/' + sweep.name
        print(f"Stopping sweep {s}")
        # Run a command to stop the sweep
        exit_code = os.system('wandb sweep --stop ' + s)
        if exit_code != 0:
            print(f"Failed to stop sweep {s}")
            print(f"Exit code: {exit_code}")
            return False
        else:
            print(f"Sweep {s} stopped successfully")
            return True
    else:
        print(f"Sweep {sweep_id} is already finished")
        return True

def make_train_sweep(wandb_project_name: str,       # string passed to the wandb.init
                        datasets: tuple,                # (train_dataset, val_dataset)
                        device: torch.device,
                        input_spec: Union[int, tuple, list],
                        num_outputs: int,
                        wandb_entity_name: Optional[str] = None,
                        metrics: dict[str, Metric] = None,
                        metrics_config: MetricsConfig = None,
                        wandb_name_prefix: Optional[str] = None):
    """
    Factory function that populates train_sweep with the right data so it can be reused for different sweeps

    :param wandb_project_name: the name of the wandb project
    :param datasets: Tuple of train and test datasets
    :param device: the device to use
    :param num_inputs: the number of inputs to the model
    :param num_outputs: the number of outputs to the model
    :param wandb_entity_name: the name of the entity to log to
    :param metrics: the metrics to be used with this sweep
    :return: populated train_sweep function
    """
    def train_sweep():
        # Step 1: Initialize a W&B run
        # The sweep controller will automatically populate wandb.config with hyperparameters
        run = wandb.init(
            entity=wandb_entity_name,
            project=wandb_project_name,
            reinit=True,  # Allow multiple runs in the same notebook
            settings=wandb.Settings(x_stats_sampling_interval=2.0, silent=True)
        )

        # DONE: Step 2: Read hyperparameters from wandb.config
        # wandb.config is like a dictionary that contains the hyperparameters for THIS run
        config = wandb.config
        print(f"wandb.config: {config}")

        # Create default config objects to pull defaults from
        default_trainer_config = TrainerConfig()
        default_model_config = ModelConfig()

        hidden_units = getattr(config, 'hidden_units', default_model_config.hidden_units)
        learning_rate = getattr(config, 'learning_rate', default_trainer_config.learning_rate)
        trainer_batch_size = getattr(config, 'trainer_batch_size', default_trainer_config.trainer_batch_size)
        evaluator_batch_size = getattr(config, 'evaluator_batch_size', default_trainer_config.evaluator_batch_size)
        dropout = getattr(config, 'dropout', default_model_config.dropout)
        num_epochs = getattr(config, 'num_epochs', default_trainer_config.num_epochs)
        momentum = getattr(config, 'momentum', default_trainer_config.momentum)
        optimizer_name = getattr(config, 'optimizer_name', default_trainer_config.optimizer_name)
        weight_decay = getattr(config, 'weight_decay', default_trainer_config.weight_decay)
        early_stopping_patience = getattr(config, 'early_stopping_patience', default_trainer_config.early_stopping_patience)
        use_scheduler = getattr(config, 'use_scheduler', default_trainer_config.use_scheduler)
        scheduler_type = getattr(config, 'scheduler_type', default_trainer_config.scheduler_type)
        scheduler_gamma = getattr(config, 'scheduler_gamma', default_trainer_config.scheduler_gamma)
        scheduler_step_size = getattr(config, 'scheduler_step_size', default_trainer_config.scheduler_step_size)
        scheduler_patience = getattr(config, 'scheduler_patience', default_trainer_config.scheduler_patience)
        scheduler_min_lr = getattr(config, 'scheduler_min_lr', default_trainer_config.scheduler_min_lr)
        model_type = getattr(config, 'model_type', default_model_config.model_type)
        in_channels = getattr(config, 'in_channels', default_model_config.in_channels)
        use_GAP = getattr(config, "use_GAP", default_model_config.use_GAP)

        # NLP / embedding
        vocab_size = getattr(config, "vocab_size", default_model_config.vocab_size)
        embedding_dim = getattr(config, "embedding_dim", default_model_config.embedding_dim)
        padding_idx = getattr(config, "padding_idx", default_model_config.padding_idx)
        freeze_embeddings = getattr(config, "freeze_embeddings", default_model_config.freeze_embeddings)
        max_seq_len = getattr(config, "max_seq_len", default_model_config.max_seq_len)
        # TextCNN1D
        num_filters = getattr(config, "num_filters", getattr(default_model_config, "num_filters", 100))
        filter_sizes = getattr(config, "filter_sizes", getattr(default_model_config, "filter_sizes", (3, 4, 5)))
        # Read all RNN fields from wandb.config using ModelConfig defaults.
        rnn_hidden_size = getattr(config, "rnn_hidden_size", default_model_config.rnn_hidden_size)
        rnn_num_layers = getattr(config, "rnn_num_layers", default_model_config.rnn_num_layers)
        bidirectional = getattr(config, "bidirectional", default_model_config.bidirectional)
        rnn_type = getattr(config, "rnn_type", default_model_config.rnn_type)
        clip_grad_norm = getattr(config, "clip_grad_norm", default_model_config.clip_grad_norm)
        # Attention
        num_heads = getattr(config, "num_heads", default_model_config.num_heads)
        # TransformerClassifier
        num_encoder_layers = getattr(config, "num_encoder_layers", default_model_config.num_encoder_layers)
        dim_feedforward = getattr(config, "dim_feedforward", default_model_config.dim_feedforward)

        # Choose loss by config ("mse" for regression, "cross_entropy" for classification).
        loss_name = getattr(config, "loss_name", "cross_entropy")

        if not isinstance(filter_sizes, tuple):
            filter_sizes = tuple(filter_sizes)

        def _parse_conv_block(raw_block):
            # Already-parsed dataclass objects are allowed
            if isinstance(raw_block, (ConvBlockConfig, ResidualBlockConfig)):
                return raw_block

            if not isinstance(raw_block, dict):
                raise TypeError(
                    f"Each conv block must be a dict or block config object, got {type(raw_block)}"
                )

            block_data = dict(raw_block)  # copy so we can safely pop
            block_type = block_data.pop("block_type", "conv")  # backward-compatible default

            if block_type == "conv":
                return ConvBlockConfig(**block_data)
            if block_type == "residual":
                return ResidualBlockConfig(**block_data)

            raise ValueError(f"Unknown block_type '{block_type}' in conv_blocks")

        # Parse the conv_blocks list
        raw_blocks = getattr(config, "conv_blocks", default_model_config.conv_blocks) or []
        conv_blocks = [_parse_conv_block(b) for b in raw_blocks]
        num_workers = getattr(config, "num_workers", default_trainer_config.num_workers)
        pin_memory = getattr(config, "pin_memory", default_trainer_config.pin_memory)

        # Set a meaningful run name based on the hyperparameters
        # This makes it much easier to identify runs in the W&B dashboard
        hidden_str = "x".join(map(str, hidden_units))  # e.g., "128x64"

        if model_type == "bow":
            run.name = f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}_vs{vocab_size}_ed{embedding_dim}_wd{weight_decay:.5f}"
        elif model_type == "textcnn":
            filter_sizes_str = "-".join(map(str, filter_sizes))
            run.name = f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}_nf{num_filters}_fs{filter_sizes_str}_wd{weight_decay:.5f}"
        elif model_type == "rnn":
            run.name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}"
                f"_hs{rnn_hidden_size}_L{rnn_num_layers}"
                f"_bi{int(bidirectional)}_{rnn_type}_wd{weight_decay:.5f}"
            )
        elif model_type == "text_attn":
            run.name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}"
                f"_ed{embedding_dim}_nh{num_heads}_wd{weight_decay:.5f}"
            )
        elif model_type == "text_transformer":
            run.name = (
                f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}"
                f"_ed{embedding_dim}_nh{num_heads}_nl{num_encoder_layers}"
                f"_dff{dim_feedforward}_wd{weight_decay:.5f}"
            )
        else:
            hidden_str = "x".join(map(str, hidden_units))
            run.name = f"{model_type}_bs{trainer_batch_size}_lr{learning_rate:.5f}_h{hidden_str}_wd{weight_decay:.5f}_m{momentum:.2f}"

        if wandb_name_prefix is not None:
            run.name = f"{wandb_name_prefix}_{run.name}"
        print(f"Run name set to: {run.name}")
        print(model_type)

        collate_fn = None
        if model_type in ("bow", "textcnn", "text_rnn", "text_attn", "text_transformer"):
            print("Using collate function")
            collate_fn = partial(text_collate_fn, max_seq_len=max_seq_len, padding_value=padding_idx)

        # Create DataLoaders using the captured train_dataset and val_dataset
        train_loader, val_loader, _ = get_dataloaders(
            train_ds=datasets[0],
            eval_ds=datasets[1],
            train_batch_size=trainer_batch_size,
            eval_batch_size=evaluator_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # DONE: Step 4: Create TrainerConfig and ModelConfig using the hyperparameters from wandb.config
        trainer_config = TrainerConfig(
            trainer_batch_size=trainer_batch_size,
            evaluator_batch_size=evaluator_batch_size,
            learning_rate=learning_rate,
            device=device,
            num_epochs=num_epochs,
            momentum=momentum,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            use_scheduler=use_scheduler,
            scheduler_type=scheduler_type,
            metrics=metrics,
            scheduler_gamma=scheduler_gamma,
            scheduler_step_size=scheduler_step_size,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            num_workers=num_workers,
            pin_memory=pin_memory,
            clip_value=clip_grad_norm
        )

        # DONE: Create a ModelConfig with hidden_units and dropout
        model_config = ModelConfig(
            model_type=model_type,
            hidden_units=hidden_units,
            dropout=dropout,
            conv_blocks=conv_blocks,
            in_channels=in_channels,
            use_GAP=use_GAP,
            # NLP / embedding parameters
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            freeze_embeddings=freeze_embeddings,
            # --- RNN support ---
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            clip_grad_norm=clip_grad_norm,
            # --- AttentionClassifier support ---
            num_heads=num_heads,
            # --- TransformerClassifier support ---
            num_encoder_layers = num_encoder_layers,
            dim_feedforward = dim_feedforward,
        )

        # DONE: Step 5: Build the model, optimizer, and criterion with build_model()
        model = build_model(
            input_spec=input_spec,
            num_outputs=num_outputs,
            config=model_config,
        )

        # DONE: Use make_optimizer() to create an optimizer using trainer_config
        optimizer = make_optimizer(model.parameters(), config=trainer_config)

        # Build criterion from the sweep config.
        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        # DONE: Step 6: Create the Trainer and train!
        # IMPORTANT: Pass run=run so the Trainer logs to THIS W&B run
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=trainer_config,
            run=run,
            metrics_config=metrics_config,
        )
        # Train the model
        results = trainer.fit(train_loader, val_loader)

        # Step 7: Clean up
        trainer.finish_run()

        print(f"✓ Run complete! Final val_loss: {results['val_loss']:.4f}, val_acc: {results['val_acc'] * 100:.2f}%")

    return train_sweep