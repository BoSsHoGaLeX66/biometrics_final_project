import numpy as np
import torch
import torchaudio

DESIRED_SAMPLE_RATE = 16000



def audio_chunk_to_tensor(chunk: np.ndarray, sample_rate: int, device: torch.device) -> torch.Tensor:
    """
    Convert a queued microphone chunk into a 16 kHz mono float tensor for inference.

    Args:
        chunk: Audio chunk from sounddevice with shape [frames] or [frames, channels].
        sample_rate: Sampling rate of the captured audio chunk.
        device: Torch device that should receive the inference tensor.

    Returns:
        Float tensor shaped [num_samples] at 16 kHz on the requested device.
    """
    tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device)
    if tensor.dim() == 2:
        tensor = tensor.mean(dim=1)
    if sample_rate != DESIRED_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, DESIRED_SAMPLE_RATE).to(device)
        tensor = resampler(tensor)
    return tensor.contiguous()
