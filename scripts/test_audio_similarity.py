"""Compare two WAV files using speaker embedding cosine similarity."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

PROJECT_ROOT = "/home/alexsearle/Documents/Bucknell/SP26/Biometrics/biometrics_final_project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

from src.my_engine.audio import wav_to_log_mel
from src.my_engine.config import ModelConfig
from src.my_engine.model import SpeakerTransformer
from src.my_engine.utils import load_model_from_checkpoint


MAX_SEQ_LENGTH = 500
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_HOMEGROWN_CHECKPOINT = Path("checkpoints/best.pt")
DEFAULT_FINETUNED_CHECKPOINT = Path("models/wavlm-speaker-id/finetune_wavlm.pt")
FINETUNED_MODEL_NAME = "microsoft/wavlm-base-plus"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model choice and input WAV paths."""
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between two speaker embeddings.",
    )
    parser.add_argument(
        "model",
        choices=("homegrown", "finetuned"),
        help="Model type to use for embedding extraction.",
    )
    parser.add_argument("first_wav", type=Path, help="Path to the first .wav file.")
    parser.add_argument("second_wav", type=Path, help="Path to the second .wav file.")
    return parser.parse_args()


def read_waveform(wav_path: Path, sr: int = DEFAULT_SAMPLE_RATE) -> torch.Tensor:
    """
    Read a WAV file and resample it to the target sample rate.

    Args:
        wav_path: Path to the input WAV file.
        sr: Target sample rate.

    Returns:
        Waveform tensor with shape [num_samples].
    """
    audio, sample_rate = sf.read(wav_path)
    waveform = torch.tensor(audio, dtype=torch.float32)

    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)

    return waveform


def wav_path_to_log_mel(wav_path: Path, sr: int = DEFAULT_SAMPLE_RATE) -> torch.Tensor:
    """
    Convert a WAV path to a time-major log-Mel spectrogram.

    Args:
        wav_path: Path to the input WAV file.
        sr: Target sample rate.

    Returns:
        Tensor with shape [time_steps, n_mels], truncated to MAX_SEQ_LENGTH.
    """
    waveform = read_waveform(wav_path, sr)
    mel = wav_to_log_mel(waveform, sr)
    mel = mel.transpose(0, 1)

    if mel.shape[0] > MAX_SEQ_LENGTH:
        return mel[:MAX_SEQ_LENGTH]
    return mel


def load_homegrown_model(device: torch.device) -> SpeakerTransformer:
    """
    Load the homegrown SpeakerTransformer checkpoint.

    Args:
        device: Device for model inference.

    Returns:
        SpeakerTransformer model in eval mode.
    """
    if DEFAULT_HOMEGROWN_CHECKPOINT.exists():
        try:
            model = load_model_from_checkpoint(DEFAULT_HOMEGROWN_CHECKPOINT, device)
        except RuntimeError:
            checkpoint = torch.load(
                DEFAULT_HOMEGROWN_CHECKPOINT,
                map_location=device,
                weights_only=False,
            )
            arch = checkpoint["model_architecture"]
            config = ModelConfig(**arch["config"])
            model = SpeakerTransformer(
                num_outputs=arch["num_outputs"],
                config=config,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(device)
    else:
        model = torch.load(
            "models/speaker_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.to(device)

    model.eval()
    return model


def homegrown_embeddings(
    model: SpeakerTransformer,
    wav_paths: tuple[Path, Path],
    device: torch.device,
) -> torch.Tensor:
    """
    Extract embeddings from the homegrown model for two WAV files.

    Args:
        model: Loaded SpeakerTransformer model.
        wav_paths: Pair of WAV file paths.
        device: Device for model inference.

    Returns:
        Tensor with shape [2, embedding_dim].
    """
    mel_specs = [wav_path_to_log_mel(path) for path in wav_paths]
    batch = pad_sequence(mel_specs, batch_first=True)
    batch = batch.permute(0, 2, 1).to(device)

    with torch.no_grad():
        return model.extract_embedding(batch)


def load_finetuned_model(
    device: torch.device,
) -> AutoModelForAudioClassification:
    """
    Load the fine-tuned WavLM audio classifier checkpoint.

    Args:
        device: Device for model inference.

    Returns:
        Fine-tuned Hugging Face audio classification model in eval mode.
    """
    state_dict = torch.load(
        DEFAULT_FINETUNED_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    num_labels = state_dict["classifier.weight"].shape[0]
    model = AutoModelForAudioClassification.from_pretrained(
        FINETUNED_MODEL_NAME,
        num_labels=num_labels,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def finetuned_embeddings(
    model: AutoModelForAudioClassification,
    wav_paths: tuple[Path, Path],
    device: torch.device,
) -> torch.Tensor:
    """
    Extract pooled WavLM hidden features for two WAV files.

    Args:
        model: Loaded fine-tuned WavLM audio classifier.
        wav_paths: Pair of WAV file paths.
        device: Device for model inference.

    Returns:
        Tensor with shape [2, hidden_size].
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(FINETUNED_MODEL_NAME)
    waveforms = [read_waveform(path).numpy() for path in wav_paths]
    inputs = feature_extractor(
        waveforms,
        sampling_rate=DEFAULT_SAMPLE_RATE,
        max_length=DEFAULT_SAMPLE_RATE * 4,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.wavlm(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
        )
        return outputs.last_hidden_state.mean(dim=1)


def cosine_similarity(embeddings: torch.Tensor) -> float:
    """
    Compute cosine similarity between the first two embedding rows.

    Args:
        embeddings: Tensor with shape [2, embedding_dim].

    Returns:
        Cosine similarity as a Python float.
    """
    normalized = F.normalize(embeddings, dim=1)
    return F.cosine_similarity(normalized[0:1], normalized[1:2]).item()


def main() -> None:
    """Load the requested model, compare the two WAV files, and print similarity."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_paths = (args.first_wav, args.second_wav)

    for wav_path in wav_paths:
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file does not exist: {wav_path}")

    if args.model == "homegrown":
        model = load_homegrown_model(device)
        embeddings = homegrown_embeddings(model, wav_paths, device)
    else:
        model = load_finetuned_model(device)
        embeddings = finetuned_embeddings(model, wav_paths, device)

    similarity = cosine_similarity(embeddings)
    print(f"Cosine similarity: {similarity:.6f}")


if __name__ == "__main__":
    main()
