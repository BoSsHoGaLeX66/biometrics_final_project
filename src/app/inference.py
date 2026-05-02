from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.my_engine.audio import wav_to_log_mel


ModelType = Literal["homegrown", "finetuned"]


def get_features(
    model: nn.Module,
    model_type: ModelType,
    tensor: torch.Tensor,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Convert an inference waveform into the correct model input and extract embeddings.

    Args:
        model: Loaded speaker recognition model.
        model_type: Model family, either "homegrown" or "finetuned".
        tensor: Raw waveform tensor for the current inference chunk.
        sample_rate: Sampling rate of the waveform tensor.

    Returns:
        Embedding features produced by the selected model.

    Raises:
        ValueError: If an unsupported model type is requested.
    """
    waveform = tensor.float()
    if waveform.dim() == 2:
        if waveform.shape[1] <= 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.mean(dim=1)
        else:
            waveform = waveform.mean(dim=0)

    log_mel = wav_to_log_mel(waveform.detach().cpu(), sample_rate).to(waveform.device)

    if model_type == "homegrown":
        return _extract_homegrown_features(model, log_mel)
    if model_type == "finetuned":
        return _extract_wavlm_features(model, waveform)
    raise ValueError(f"Unsupported model type: {model_type}")


def _extract_homegrown_features(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract normalized speaker embeddings from the homegrown transformer model.

    Args:
        model: Loaded homegrown speaker transformer model.
        tensor: Log-Mel spectrogram tensor shaped [n_mels, time_steps] or batched equivalent.

    Returns:
        Speaker embedding tensor produced by the homegrown model.
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        if hasattr(model, "extract_embedding"):
            return model.extract_embedding(tensor)
        logits = model(tensor)
        return F.normalize(logits, dim=-1)


def _extract_wavlm_features(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract projected utterance embeddings from a fine-tuned WavLM model.

    Args:
        model: Loaded WavLM sequence classification model.
        tensor: Raw waveform tensor shaped [num_samples] or [batch_size, num_samples].

    Returns:
        Normalized projected WavLM embedding tensor.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2 and tensor.shape[-1] == 1:
        tensor = tensor.transpose(0, 1)

    with torch.no_grad():
        wavlm_outputs = model.wavlm(tensor)
        pooled = wavlm_outputs.last_hidden_state.mean(dim=1)
        projected = model.projector(pooled)
        return F.normalize(projected, dim=-1)
