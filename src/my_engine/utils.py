import torch
import torch.nn as nn
from typing import Tuple, Union, Optional, List
from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np
from src.my_engine.config import (
    ModelConfig,
    TrainerConfig,
    ConvBlockConfig,
    ResidualBlockConfig,
)
from .model import MLP_Model, CNN_Model, BagOfEmbeddings, TextCNN1D, RNNModel, TextRNNModel, AttentionClassifier, TransformerClassifier, SpeakerTransformer
"""
utils.py

This module contains a collection of helper utility functions that will be used throughout the course.
You will find reusable functions for metrics, data handling, and other tools to support labs, 
assignments, and projects.

Course: CSCI 357 - AI and Neural Networks
Author: Alex Searle
Date: 02/17/2026 

"""

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # DONE: Implement the accuracy_from_logits function.
    # If logits are not a 1-D tensor
    if logits.shape[1] != 1:
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()
    # If logits are a 1-D tensor
    else:
        logits = logits.squeeze()
        logits = nn.Sigmoid()(logits)
        preds = torch.where(logits > 0.5, 1, 0)
        return (preds == labels).float().mean().item()

def build_model(input_spec: Union[int, List[int]], num_outputs: int, config: ModelConfig) -> nn.Module:
    """
    This function builds and returns the model that is specified by the parameters
    :param input_spec: The number of inputs to the model or the image size if building CNN
    :param num_outputs: the number of outputs for the model
    :param config: the model config that specifies how to build the model
    :return: a model built to specification
    """
    if config.model_type == "mlp":
        if not isinstance(input_spec, int):
            raise ValueError("MLP requires input_spec as int (flattened input size).")
        return MLP_Model(num_inputs=input_spec, num_outputs=num_outputs, config=config)
    elif config.model_type == "cnn":
        if not isinstance(input_spec, (tuple, list)) or len(input_spec) != 2:
            raise ValueError("CNN requires input_spec as (height, width).")
        h, w = input_spec[0], input_spec[1]
        return CNN_Model(input_height=h, input_width=w, num_outputs=num_outputs, config=config)
    elif config.model_type == "bow":
        # DONE: Instantiate and return a BagOfEmbeddings model.
        #       input_spec is unused here; vocab_size lives in config.
        return BagOfEmbeddings(num_outputs=num_outputs, config=config)
    elif config.model_type == "textcnn":
        return TextCNN1D(
            num_outputs=num_outputs,
            config=config,
        )
    elif config.model_type == "rnn":
        if not isinstance(input_spec, int) or input_spec <= 0:
            raise ValueError("RNN requires input_spec as int > 0 (input_size = features per time step).")
        return RNNModel(input_size=input_spec, num_outputs=num_outputs, config=config)
    elif config.model_type == "text_rnn":
        return TextRNNModel(num_outputs=num_outputs, config=config)
    elif config.model_type == "text_attn":
        return AttentionClassifier(num_outputs=num_outputs, config=config)
    elif config.model_type == "text_transformer":
        return TransformerClassifier(num_outputs=num_outputs, config=config)
    elif config.model_type == "speaker_transformer":
        return SpeakerTransformer(num_outputs=num_outputs, config=config)
    else:
        raise ValueError(f"Model type '{config.model_type}' not supported. Supported types: 'mlp', 'cnn', 'bow', 'textcnn', 'rnn', 'text_rnn', 'text_attn', 'text_transformer', 'speaker_transformer'")

def make_optimizer(params, config: TrainerConfig) -> torch.optim.Optimizer:
    """
    Factory for optimizers.

    Args:
        params (Iterable[torch.nn.Parameter]): Parameters to optimize.
        config (TrainerConfig): Configuration for the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """

    if config.optimizer_name.lower() == "sgd":
        # DONE: Return vanilla SGD (no momentum)
        return torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_name.lower() == "momentum":
        # DONE: Return SGD with momentum (typically 0.9)
        return torch.optim.SGD(params, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer_name.lower() == "adam":
        # DONE: Return Adam (adaptive learning rates per parameter)
        return torch.optim.Adam(params, weight_decay=config.weight_decay, lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_name}")

def load_model_from_checkpoint(checkpoint_path: Union[str, Path],
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """Reconstructs any model from a checkpoint file.

    This factory function inspects the checkpoint's model_architecture to
    determine the model type, then dispatches to the appropriate constructor.

    NOTE: This ONLY restores the model architecture and weights, not optimizer state or other metadata.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load onto (default: CPU)

    Returns:
        Reconstructed model with loaded weights

    Raises:
        ValueError: If model_type in checkpoint is unrecognized
        FileNotFoundError: if the checkpoint file does not exist
        KeyError: if the checkpoint is missing the model_architecture metadata
    """
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError("Checkpoint file does not exist")

    arch = checkpoint['model_architecture']

    if not arch:
        raise KeyError("No model_architecture metadata found in checkpoint file")

    model_type = arch.get("model_type")

    if model_type is None:
        model_class = arch.get("model_class")
        dict_class_to_type = {
            "MLP_Model": "mlp",
            "CNN_Model": "cnn",
            "BagOfEmbeddings": "bow",
            "TextCNN1D": "textcnn",
            "RNNModel": "rnn",
            "TextRNNModel": "text_rnn",
            "AttentionClassifier": "attn_text",
            "SpeakerTransformer": "speaker_transformer"
        }
        model_type = dict_class_to_type.get(model_class)

    if model_type is None:
        raise ValueError(
            "Could not determine model type from checkpoint architecture metadata"
        )

    config = _rebuild_model_config(arch["config"])

    if model_type == "mlp":
        model = MLP_Model(
            num_inputs=arch["num_inputs"],
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "cnn":
        model = CNN_Model(
            input_height=arch["input_height"],
            input_width=arch["input_width"],
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "bow":
        model = BagOfEmbeddings(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "textcnn":
        model = TextCNN1D(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "rnn":
        model = RNNModel(
            input_size=arch["input_size"],
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif config.model_type == "text_rnn":
        model = TextRNNModel(num_outputs=arch["num_outputs"], config=config)
    elif model_type == "text_attn":
        model = AttentionClassifier(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "text_transformer":
        model = TransformerClassifier(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    elif model_type == "speaker_transformer":
        model = SpeakerTransformer(
            num_outputs=arch["num_outputs"],
            config=config,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def _rebuild_model_config(config_dict: dict) -> ModelConfig:
    """Rehydrate ModelConfig from checkpoint-safe dictionaries."""
    cfg = dict(config_dict)

    if "conv_blocks" in cfg and isinstance(cfg["conv_blocks"], list):
        rebuilt_blocks = []
        for block in cfg["conv_blocks"]:
            if isinstance(block, dict):
                block_type = block.get("block_type", "conv")
                block_payload = {k: v for k, v in block.items() if k != "block_type"}
                if block_type == "residual":
                    rebuilt_blocks.append(ResidualBlockConfig(**block_payload))
                else:
                    rebuilt_blocks.append(ConvBlockConfig(**block_payload))
            else:
                rebuilt_blocks.append(block)
        cfg["conv_blocks"] = rebuilt_blocks

    return ModelConfig(**cfg)

def make_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Factory for learning rate schedulers.

    Args:
        optimizer: The optimizer to schedule
        config: Configuration containing scheduler settings

    Returns:
        Scheduler instance, or None if use_scheduler is False

    Raises:
        ValueError: If scheduler_type is unrecognized
    """
    if not config.use_scheduler:
        return None

    if config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
    # Add more schedulers as needed...
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

def get_preds(logits: torch.Tensor) -> torch.Tensor:
    # If logits are not a 1-D tensor
    if logits.shape[1] != 1:
        preds = torch.argmax(logits, dim=1)
    # If logits are a 1-D tensor
    else:
        logits = logits.squeeze()
        logits = nn.Sigmoid()(logits)
        preds = torch.where(logits > 0.5, 1, 0).squeeze()

    return preds

def lr_range_test(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_iterations: int = 100,
) -> tuple[list[float], list[float]]:
    """Performs Leslie Smith's LR Range Test.

    Trains the model for num_iterations mini-batches, exponentially increasing the
    learning rate from start_lr to end_lr. Records the loss at each step.

    WARNING: This function modifies the model weights and optimizer state.
    You should create a fresh model/optimizer before calling this, or save and
    restore a checkpoint afterward.

    Args:
        model: The model to test.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer (will have its LR modified).
        device: Device to train on.
        start_lr: Starting learning rate.
        end_lr: Ending learning rate.
        num_iterations: Number of mini-batches to train.

    Returns:
        Tuple of (lrs, losses) -- lists of learning rates and corresponding losses.
    """
    # Implement the LR range test.
    # 1. Set the optimizer's LR to start_lr
    # 2. Compute the multiplicative factor: gamma = (end_lr / start_lr) ** (1 / num_iterations)
    # 3. Create a LambdaLR scheduler that multiplies LR by gamma each step:
    #    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: gamma ** step)
    # 4. Loop for num_iterations batches (cycle through train_loader if needed):
    #    a. Get the next batch, move to device
    #    b. Forward pass, compute loss
    #    c. Record current LR and loss
    #    d. Backward pass, optimizer step, scheduler step
    #    e. If loss > 4 * best_loss_so_far, stop early (diverging)
    # 5. Return (lrs, losses)

    # Move the model to the specified device and set it to training mode
    model = model.to(device)
    model.train()

    # Set the optimizer's learning rate to the starting value
    for pg in optimizer.param_groups:
        pg["lr"] = start_lr

    # Calculate the multiplicative factor for exponentially increasing the LR
    gamma = (end_lr / start_lr) ** (1 / num_iterations)

    # Create a LambdaLR scheduler that multiplies the LR by gamma each step
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: gamma ** step
    )

    lrs: list[float] = []     # Store the learning rates for each batch
    losses: list[float] = []  # Store the losses for each batch
    best_loss = float("inf")  # Track the smallest loss encountered
    loader_iter = iter(train_loader)  # Iterator for cycling through the train loader

    for _ in range(num_iterations):
        try:
            # Get the next batch from the data loader
            inputs, targets = next(loader_iter)
        except StopIteration:
            # Restart the iterator if we run out of data
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        # Move the data to the correct device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero out the previous gradients
        optimizer.zero_grad()

        # Forward pass: compute outputs and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_val = loss.item()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Record the current learning rate and loss
        lrs.append(curr_lr)
        losses.append(loss_val)

        # Update the best loss seen so far, and early stop if loss diverges
        if loss_val < best_loss:
            best_loss = loss_val
        elif loss_val > 4 * best_loss:
            break

        # Backward pass and optimizer/scheduler step
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Return recorded learning rates and corresponding losses
    return lrs, losses

def compute_confusion_matrix(model, data_loader, device):
    """Compute a confusion matrix by running the model over the entire dataset.

    Args:
        model: Trained model.
        data_loader: DataLoader for the evaluation set.
        device: Device to run inference on.

    Returns:
        Tuple of (confusion_matrix as numpy array, all_preds list, all_labels list)
    """
    # DONE: Compute the list of predictions and actual labels for every example in the data_loader passed in
    # NOTE: Be sure to account for the device, but make sure actual predictions are on the CPU
    # NOTE: Be sure to check if the model is in training mode, and if so, set it to evaluation mode, then back before returning
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            preds = model(X)
            preds = preds.argmax(dim=1)
            preds = preds.to('cpu')
            all_preds.extend(preds.tolist())
            y = y.to('cpu')
            all_labels.extend(y.tolist())


    cm = confusion_matrix(all_labels, all_preds)


    return cm, all_preds, all_labels

def compute_saliency_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None,
    device: torch.device = None,
) -> tuple[np.ndarray, int]:
    """Compute a saliency map for a single image.

    The saliency map highlights which pixels most influence the model's
    prediction by computing |d(score_c) / d(input)|.

    Args:
        model: Trained model (will be set to eval mode).
        image_tensor: Single image tensor of shape (C, H, W). Do NOT include batch dim.
        target_class: Class index to compute saliency for.
                      If None, uses the model's predicted class (argmax).
        device: Device for computation. If None, uses the model's device.

    Returns:
        Tuple of (saliency_map as 2D numpy array normalized to [0, 1],
                  target_class index used).
    """
    # DONE: Set the model to evaluation mode for correct inference
    model.eval()
    # DONE: If device is not specified, use the model's device for computation
    if device is None:
        device = next(model.parameters()).device

    # DONE: Prepare the input image
    #   - Add batch dimension to image_tensor
    #   - Move image to the correct device
    #   - Ensure image.requires_grad is True
    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.requires_grad_(True)

    # DONE: Forward pass through the model to get the output logits
    logits = model.forward(image_tensor)

    # DONE: Pick the target class to compute the saliency map for
    #   - If target_class is None, use model's predicted class (argmax)
    if target_class is None:
        target_class = logits.argmax(dim=1)

    # DONE: Compute the score for the selected class
    #   - Backpropagate the gradient from this score
    target = logits[0, target_class]
    target.backward()

    # DONE: Extract the gradient (saliency) from image.grad
    #   - Take the absolute value
    saliency = image_tensor.grad.abs()

    # DONE: For color images (multiple channels), take the max across channels
    if image_tensor.shape[1] != 1:
        saliency = saliency.squeeze(0).max(dim=0).values

    # DONE: Normalize the saliency map to [0, 1]
    sal_max = torch.max(saliency)
    sal_min = torch.min(saliency)
    saliency = (saliency - sal_min) / (sal_max - sal_min)
    # DONE: Return the normalized saliency map and the target_class used
    saliency = saliency.detach().cpu().numpy()


    return saliency, target_class

def denormalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Denormalizes a tensor image using the given mean and standard deviation values.

    Args:
        tensor (torch.Tensor): The image tensor to be denormalized. Expected shape: (C, H, W).
        mean (tuple, optional): Mean values for each channel. Default is (0.5, 0.5, 0.5).
        std (tuple, optional): Standard deviation values for each channel. Default is (0.5, 0.5, 0.5).

    Returns:
        torch.Tensor: The denormalized image tensor, with values clipped to [0, 1].
    """

    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    mean = torch.tensor(mean)
    return std * tensor + mean


def test_eval(model, test_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs through a test loader and collects all the preds and lables
    :param model: nn.Module
    :param test_loader: DataLoader
    :param device: device of the model
    :return: all_preds and all_lables
    """
    all_preds = []
    all_labels = []
    for X, y in test_loader:
        X = X.to(device)

        logits = model(X)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_labels