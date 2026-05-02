from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import WavLMConfig, WavLMForSequenceClassification


ModelType = Literal["homegrown", "finetuned"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOMEGROWN_MODEL_PATH = PROJECT_ROOT / "models" / "speaker_model.pt"
FINETUNED_MODEL_PATH = PROJECT_ROOT / "models" / "wavlm-speaker-id" / "finetune_wavlm.pt"


def load_model(model_type: ModelType, device: torch.device | str = "cpu") -> nn.Module:
    """
    Load the requested speaker recognition model for inference.

    Args:
        model_type: Model family to load, either "homegrown" or "finetuned".
        device: Torch device where the model should be placed.

    Returns:
        Loaded model in evaluation mode.

    Raises:
        ValueError: If an unsupported model type is requested.
    """
    if model_type == "homegrown":
        return load_homegrown_model(device=device)
    if model_type == "finetuned":
        return load_finetuned_model(device=device)
    raise ValueError(f"Unsupported model type: {model_type}")


def load_homegrown_model(
    model_path: Path = HOMEGROWN_MODEL_PATH,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Load the serialized homegrown speaker transformer model.

    Args:
        model_path: Path to the saved PyTorch model file.
        device: Torch device where the model should be placed.

    Returns:
        Homegrown speaker model in evaluation mode.
    """
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def load_finetuned_model(
    model_path: Path = FINETUNED_MODEL_PATH,
    device: torch.device | str = "cpu",
) -> WavLMForSequenceClassification:
    """
    Load the fine-tuned WavLM speaker identification model from a state dict.

    Args:
        model_path: Path to the fine-tuned WavLM state dictionary.
        device: Torch device where the model should be placed.

    Returns:
        WavLM sequence classification model in evaluation mode.
    """
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    num_labels = state_dict["classifier.weight"].shape[0]
    config = WavLMConfig(num_labels=num_labels)
    model = WavLMForSequenceClassification(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
