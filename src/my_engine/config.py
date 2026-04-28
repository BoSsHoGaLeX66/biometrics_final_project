
"""
config.py

This module contains configuration dataclasses useful for AI and ML projects.

Course: CSCI 357 - AI and Neural Networks
Author: Alex Searle
Date: 02/17/2026

"""

from dataclasses import dataclass, field
import torch
from typing import List, Optional, Union
from torchmetrics import Metric


@dataclass
class MetricsConfig:
    task: str = 'binary'
    names: List[str] = field(default_factory=lambda: ['accuracy'])
    num_classes: int = 2
    average: str = 'macro'

@dataclass
class TrainerConfig:
    trainer_batch_size: int = 64
    evaluator_batch_size: int = 256
    learning_rate: float = 0.001
    device: torch.device = torch.device("cpu")
    num_epochs: int = 10
    weight_decay: float = 0.0
    early_stopping_patience: Optional[int] = 5          # Set to None to disable early stopping
    early_stopping_min_delta: float = 0.001
    momentum: float = 0.9
    optimizer_name: str = "Adam"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_last_filename = "last.pt"
    checkpoint_save_interval: int = 5
    checkpoint_best_filename: str = "best.pt"

    # Learning Rate Scheduler Settings
    use_scheduler: bool = False  # Enable/disable scheduling
    scheduler_type: str = "reduce_on_plateau"  # Options: "step", "exponential", "cosine", "reduce_on_plateau"
    scheduler_step_size: int = 10  # For StepLR: epochs between LR drops
    scheduler_gamma: float = 0.1  # Factor to reduce LR
    scheduler_patience: int = 3  # For ReduceLROnPlateau: epochs to wait before reducing
    scheduler_min_lr: float = 1e-6  # Minimum learning rate (prevents it from going too low)

    # Torch metrics to use
    metrics: dict[str, Metric] = None # A dictionary with the name and then the function of the torch metric

    # DataLoader settings
    num_workers: int = 2
    pin_memory: bool = True
    clip_value: float = 0

@dataclass
class ConvBlockConfig:
    """Configuration for a single [Conv2d -> ReLU -> MaxPool2d] block.

    Attributes:
        out_channels: Number of filters (output feature maps) for this block.
        kernel_size: Spatial size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        pool_size: Kernel size for MaxPool2d. Set to 0 to skip pooling.
    """
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool_size: int = 2
    batch_norm: bool = False  # Defaults to False for backwards compatability


@dataclass
class ResidualBlockConfig:
    """
    Configuration for a single residual block.

    This configuration specifies the parameters used to construct a residual block within a convolutional neural network.
    It determines the number of output channels (filters) and the stride for the convolution, enabling the creation
    of blocks that support skip connections and flexible downsampling, as used in deep residual networks.
    NOTE: Remember - pool_size is intentionally omitted - stride handles downsampling

    Attributes:
        out_channels: Number of filters (output feature maps) for this block
        stride: Stride of the convolution.
    """
    out_channels: int
    stride: int = 1

@dataclass
class ModelConfig:
    """
    Configuration for various neural network architectures (MLP, CNN, RNN, TextCNN).

    Attributes:
        model_type: The type of model to instantiate (e.g., "mlp", "cnn", "textcnn1d", "rnn", "text_rnn", "text_attn", "text_transformer").
        hidden_units: List of neurons for each fully connected hidden layer.
        dropout: List of dropout probabilities for each hidden layer.
        conv_blocks: List of configuration for convolutional or residual blocks.
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        use_GAP: If True, uses Global Average Pooling before the final classification layer.
        vocab_size: Total number of tokens in the vocabulary (for NLP tasks).
        embedding_dim: Dimensionality of token embeddings.
        padding_idx: Index used for padding tokens in the vocabulary.
        freeze_embeddings: If True, embedding weights will not be updated during training.
        filter_sizes: Sizes of filters for the TextCNN1D model.
        num_filters: Number of filters per size for the TextCNN1D model.
        rnn_hidden_size: Hidden state dimensionality for RNN layers.
        rnn_num_layers: Number of stacked RNN layers.
        bidirectional: If True, uses a bidirectional RNN architecture.
        rnn_type: Type of recurrent unit ("rnn", "lstm", "gru").
        clip_grad_norm: Maximum gradient norm for clipping; 0.0 disables clipping.
        num_heads: Number of attention heads (used by text_attn). Must stisfy embed_dim % num_heads == 0
        --- Transformer encoder fields ---
        num_encoder_layers: Number of stacked encoder layers (used by text_transformer).
        dim_feedforward: FFN hidden dimension (used by text_transformer). Typically 4 * embedding_dim.
    """
    model_type: str = "mlp"
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2])
    # CNN fields
    conv_blocks: List[Union[ConvBlockConfig, ResidualBlockConfig]] = field(default_factory=list)
    in_channels: int = 1  # 1 for grayscale, 3 for RGB
    use_GAP: bool = False
    # --- NLP / Embedding fields ---
    vocab_size: int = 0
    embedding_dim: int = 100
    padding_idx: int = 0
    freeze_embeddings: bool = False
    max_seq_len: Optional[int] = None   # Truncate input sequences to at most this many tokens.
                                        # None = no truncation. Critical for text_attn, whose
                                        # attention score matrix scales as O(batch * L^2).
    # --- TextCNN1D fields ---
    filter_sizes: List[int] = field(
        default_factory=lambda: [3, 4, 5])  # The sizes of the filters to use in the TextCNN1D model.
    num_filters: int = 100  # The number of filters to use in the TextCNN1D model.
    # --- RNN fields ---
    rnn_hidden_size: int = 64  # Hidden state dimensionality for RNN layers
    rnn_num_layers: int = 1  # Number of stacked RNN layers
    bidirectional: bool = False  # If True, use bidirectional RNN
    rnn_type: str = "rnn"  # "rnn" for vanilla RNN, "lstm" for LSTM, "gru" for GRU
    clip_grad_norm: float = 0.0  # Max gradient norm for clipping (0 = disabled)
    # --- Attention fields ---
    num_heads: int = 4
    # --- Transformer encoder fields ---
    num_encoder_layers: int = 2  # Number of stacked TransformerEncoderLayers
    dim_feedforward: int = 512  # Hidden dimension in the FFN sublayer (typically 4 * embedding_dim)
    # --- Speaker Transformer fields ---
    n_mels: int = 80  # Number of Mel bands in the input spectrogram


