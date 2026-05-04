"""Create genuine and imposter speaker similarity distributions from WAV data."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.app.inference import (
    _extract_homegrown_features,
    _extract_wavlm_features as _extract_finetuned_features,
)
from src.app.load import load_model
from src.my_engine.audio import wav_to_log_mel


ModelType = Literal["homegrown", "finetuned"]
DEFAULT_SAMPLE_RATE = 16_000
MAX_SEQ_LENGTH = 500


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model selection and graph generation.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create cosine similarity arrays and histograms for speaker verification.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("homegrown", "finetuned"),
        help="Model type to use for embedding extraction.",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "wav",
        help="Directory containing id1* speaker folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "graphs",
        help="Directory where arrays and plots will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for speaker selection.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target audio sample rate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of WAV files per model batch.")
    parser.add_argument("--genuine-users", type=int, default=10, help="Number of genuine speakers to sample.")
    parser.add_argument("--imposter-users", type=int, default=20, help="Number of imposter speakers to sample.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for model inference.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram generation and only save the similarity arrays.",
    )
    return parser.parse_args()


def get_user_wav_paths(wav_dir: Path) -> dict[str, list[Path]]:
    """
    Load all WAV paths grouped by speaker ID folder.

    Args:
        wav_dir: Directory containing one folder per user.

    Returns:
        Mapping from user folder name to sorted WAV file paths.

    Raises:
        FileNotFoundError: If the WAV directory does not exist.
    """
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory does not exist: {wav_dir}")

    user_wavs: dict[str, list[Path]] = {}
    for user_dir in sorted(wav_dir.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.startswith("id1"):
            continue

        wav_paths = sorted(user_dir.rglob("*.wav"))
        if wav_paths:
            user_wavs[user_dir.name] = wav_paths

    return user_wavs


def select_users(
    user_wavs: dict[str, list[Path]],
    genuine_count: int,
    imposter_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Randomly select disjoint genuine and imposter users.

    Args:
        user_wavs: Mapping from user ID to WAV paths.
        genuine_count: Number of genuine users to select.
        imposter_count: Number of imposter users to select.
        seed: Random seed for reproducibility.

    Returns:
        Two lists containing genuine user IDs and imposter user IDs.

    Raises:
        ValueError: If there are not enough users with WAV files.
    """
    users = sorted(user_wavs)
    needed_users = genuine_count + imposter_count
    if len(users) < needed_users:
        raise ValueError(f"Need {needed_users} users with WAV files, found {len(users)}.")

    rng = random.Random(seed)
    selected = rng.sample(users, needed_users)
    return selected[:genuine_count], selected[genuine_count:]


def read_waveform(wav_path: Path, sample_rate: int) -> torch.Tensor:
    """
    Read one WAV file as a mono waveform at the target sample rate.

    Args:
        wav_path: Path to the WAV file.
        sample_rate: Target sampling rate.

    Returns:
        Mono waveform tensor shaped [num_samples].
    """
    audio, file_sample_rate = sf.read(wav_path)
    waveform = torch.tensor(audio, dtype=torch.float32)

    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)

    if file_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)

    return waveform


def waveform_to_log_mel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Convert a waveform to a time-major log-Mel spectrogram for the homegrown model.

    Args:
        waveform: Mono waveform tensor.
        sample_rate: Sampling rate for feature extraction.

    Returns:
        Log-Mel tensor shaped [time_steps, n_mels].
    """
    log_mel = wav_to_log_mel(waveform, sample_rate).transpose(0, 1)
    if log_mel.shape[0] > MAX_SEQ_LENGTH:
        return log_mel[:MAX_SEQ_LENGTH]
    return log_mel


def extract_embeddings(
    model: nn.Module,
    model_type: ModelType,
    wav_paths: list[Path],
    sample_rate: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract speaker embeddings from WAV files in model batches.

    Args:
        model: Loaded speaker recognition model.
        model_type: Model family to use for preprocessing and inference.
        wav_paths: WAV files to embed.
        sample_rate: Target audio sample rate.
        batch_size: Number of WAV files per model batch.
        device: Torch device for inference.

    Returns:
        Normalized embedding tensor shaped [num_wavs, embedding_dim] on CPU.
    """
    validate_batch_size(batch_size)

    if model_type == "homegrown":
        return extract_homegrown_embeddings(model, wav_paths, sample_rate, batch_size, device)
    return extract_finetuned_embeddings(model, wav_paths, sample_rate, batch_size, device)


def validate_batch_size(batch_size: int) -> None:
    """
    Validate the requested model batch size.

    Args:
        batch_size: Number of examples per model batch.

    Raises:
        ValueError: If batch_size is less than one.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")


def extract_homegrown_embeddings(
    model: nn.Module,
    wav_paths: list[Path],
    sample_rate: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract homegrown model embeddings with a batched DataLoader.

    Args:
        model: Loaded homegrown speaker model.
        wav_paths: WAV files to embed.
        sample_rate: Target sample rate.
        batch_size: Number of examples per model batch.
        device: Torch device for inference.

    Returns:
        Embedding tensor shaped [num_wavs, embedding_dim] on CPU.
    """
    mel_specs = [waveform_to_log_mel(read_waveform(path, sample_rate), sample_rate) for path in wav_paths]
    if not mel_specs:
        return torch.empty(0)

    dataloader = DataLoader(
        mel_specs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_homegrown_batch,
    )

    embeddings: list[torch.Tensor] = []
    for batch in dataloader:
        batch = batch.to(device)
        batch_embeddings = _extract_homegrown_features(model, batch)
        embeddings.append(batch_embeddings.detach().cpu())

    return torch.vstack(embeddings)


def extract_finetuned_embeddings(
    model: nn.Module,
    wav_paths: list[Path],
    sample_rate: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract fine-tuned WavLM embeddings with a batched DataLoader.

    Args:
        model: Loaded fine-tuned WavLM model.
        wav_paths: WAV files to embed.
        sample_rate: Target sample rate.
        batch_size: Number of examples per model batch.
        device: Torch device for inference.

    Returns:
        Embedding tensor shaped [num_wavs, embedding_dim] on CPU.
    """
    waveforms = [read_waveform(path, sample_rate) for path in wav_paths]
    if not waveforms:
        return torch.empty(0)

    dataloader = DataLoader(
        waveforms,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_finetuned_batch,
    )

    embeddings: list[torch.Tensor] = []
    for batch in dataloader:
        batch = batch.to(device)
        batch_embeddings = _extract_finetuned_features(model, batch)
        embeddings.append(batch_embeddings.detach().cpu())

    return torch.vstack(embeddings)


def collate_homegrown_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Pad log-Mel spectrograms into a homegrown model batch.

    Args:
        batch: Time-major log-Mel tensors shaped [time_steps, n_mels].

    Returns:
        Batch tensor shaped [batch_size, n_mels, padded_time_steps].
    """
    return pad_sequence(batch, batch_first=True).permute(0, 2, 1)


def collate_finetuned_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Pad waveforms into a fine-tuned WavLM model batch.

    Args:
        batch: Waveform tensors shaped [num_samples].

    Returns:
        Batch tensor shaped [batch_size, padded_num_samples].
    """
    return pad_sequence(batch, batch_first=True)


def flatten_selected_paths(user_wavs: dict[str, list[Path]], users: list[str]) -> tuple[list[Path], list[str]]:
    """
    Flatten selected user WAV paths while retaining speaker labels.

    Args:
        user_wavs: Mapping from user ID to WAV paths.
        users: Ordered user IDs to flatten.

    Returns:
        Tuple containing WAV paths and aligned user labels.
    """
    wav_paths: list[Path] = []
    labels: list[str] = []
    for user in users:
        paths = user_wavs[user]
        wav_paths.extend(paths)
        labels.extend([user] * len(paths))
    return wav_paths, labels


def compute_similarity_arrays(
    genuine_embeddings: torch.Tensor,
    genuine_labels: list[str],
    imposter_embeddings: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute genuine same-speaker and imposter cross-speaker cosine similarities.

    Args:
        genuine_embeddings: Embeddings for selected genuine speakers.
        genuine_labels: Speaker labels aligned with genuine_embeddings.
        imposter_embeddings: Embeddings for selected imposter speakers.

    Returns:
        Pair of NumPy arrays: genuine similarities and imposter similarities.
    """
    normalized_genuine = F.normalize(genuine_embeddings.float(), dim=1)
    normalized_imposters = F.normalize(imposter_embeddings.float(), dim=1)

    genuine_similarity_list: list[float] = []
    user_indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(genuine_labels):
        user_indices[label].append(index)

    for indices in user_indices.values():
        if len(indices) < 2:
            continue
        speaker_embeddings = normalized_genuine[indices]
        speaker_similarity = speaker_embeddings @ speaker_embeddings.T
        row_indices, column_indices = torch.triu_indices(len(indices), len(indices), offset=1)
        genuine_similarity_list.extend(speaker_similarity[row_indices, column_indices].tolist())

    imposter_similarity_matrix = normalized_genuine @ normalized_imposters.T
    imposter_similarity_list = imposter_similarity_matrix.reshape(-1).tolist()

    genuine_similarities = np.asarray(genuine_similarity_list, dtype=np.float32)
    imposter_similarities = np.asarray(imposter_similarity_list, dtype=np.float32)
    return genuine_similarities, imposter_similarities


def save_similarity_arrays(
    output_dir: Path,
    model_type: ModelType,
    genuine_similarities: np.ndarray,
    imposter_similarities: np.ndarray,
    genuine_users: list[str],
    imposter_users: list[str],
) -> Path:
    """
    Save similarity arrays and selected user IDs to an NPZ file.

    Args:
        output_dir: Destination directory.
        model_type: Model family used for inference.
        genuine_similarities: Same-speaker similarity array.
        imposter_similarities: Genuine-vs-imposter similarity array.
        genuine_users: Selected genuine user IDs.
        imposter_users: Selected imposter user IDs.

    Returns:
        Path to the saved NPZ file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_type}_similarities.npz"
    np.savez(
        output_path,
        genuine_similarities=genuine_similarities,
        imposter_similarities=imposter_similarities,
        genuine_users=np.asarray(genuine_users),
        imposter_users=np.asarray(imposter_users),
    )
    return output_path


def plot_similarity_histogram(
    output_dir: Path,
    model_type: ModelType,
    genuine_similarities: np.ndarray,
    imposter_similarities: np.ndarray,
) -> Path:
    """
    Plot genuine and imposter cosine similarity histograms.

    Args:
        output_dir: Destination directory.
        model_type: Model family used for inference.
        genuine_similarities: Same-speaker similarity array.
        imposter_similarities: Genuine-vs-imposter similarity array.

    Returns:
        Path to the saved plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib.pyplot as plt

    output_path = output_dir / f"{model_type}_similarity_histogram.png"

    plt.figure(figsize=(10, 6))
    plt.hist(imposter_similarities, bins=60, alpha=0.65, label="Imposter", density=True)
    plt.hist(genuine_similarities, bins=60, alpha=0.65, label="Genuine", density=True)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"{model_type.title()} speaker verification similarities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_ROC_curve(
    output_dir: Path,
    model_type: ModelType,
    genuine_similarities: np.ndarray,
    imposter_similarities: np.ndarray,
) -> Path:
    """
    Plot genuine accept rate against imposter accept rate across thresholds.

    Args:
        output_dir: Destination directory.
        model_type: Model family used for inference.
        genuine_similarities: Same-speaker similarity array.
        imposter_similarities: Genuine-vs-imposter similarity array.

    Returns:
        Path to the saved ROC curve.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib.pyplot as plt

    thresholds, false_accept_rates, false_reject_rates = calculate_error_rates(
        genuine_similarities,
        imposter_similarities,
    )
    genuine_accept_rates = 1.0 - false_reject_rates
    eer_index, eer = find_equal_error_rate(false_accept_rates, false_reject_rates)
    eer_threshold = thresholds[eer_index]
    eer_far = false_accept_rates[eer_index]

    output_path = output_dir / f"{model_type}_roc_curve.png"
    plt.figure(figsize=(8, 8))
    plt.plot(false_accept_rates, genuine_accept_rates, linewidth=2, label="ROC")
    plt.axvline(
        eer_far,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"EER = {eer:.3f} at threshold {eer_threshold:.3f}",
    )
    plt.scatter([eer_far], [1.0 - false_reject_rates[eer_index]], color="black", zorder=3)
    plt.xlabel("Imposter accept rate (FAR)")
    plt.ylabel("Genuine accept rate")
    plt.title(f"{model_type.title()} ROC curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_DET_curve(
    output_dir: Path,
    model_type: ModelType,
    genuine_similarities: np.ndarray,
    imposter_similarities: np.ndarray,
) -> Path:
    """
    Plot false negative rate against false positive rate across thresholds.

    Args:
        output_dir: Destination directory.
        model_type: Model family used for inference.
        genuine_similarities: Same-speaker similarity array.
        imposter_similarities: Genuine-vs-imposter similarity array.

    Returns:
        Path to the saved DET curve.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib.pyplot as plt

    thresholds, false_positive_rates, false_negative_rates = calculate_error_rates(
        genuine_similarities,
        imposter_similarities,
    )
    eer_index, eer = find_equal_error_rate(false_positive_rates, false_negative_rates)
    eer_threshold = thresholds[eer_index]
    eer_fpr = false_positive_rates[eer_index]

    output_path = output_dir / f"{model_type}_det_curve.png"
    plt.figure(figsize=(8, 8))
    plt.plot(false_positive_rates, false_negative_rates, linewidth=2, label="DET")
    plt.axvline(
        eer_fpr,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"EER = {eer:.3f} at threshold {eer_threshold:.3f}",
    )
    plt.scatter([eer_fpr], [false_negative_rates[eer_index]], color="black", zorder=3)
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("False negative rate (FNR)")
    plt.title(f"{model_type.title()} DET curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

def plot_FAR_FRR_curve(
    output_dir: Path,
    model_type: ModelType,
    genuine_similarities: np.ndarray,
    imposter_similarities: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib.pyplot as plt

    thresholds, false_positive_rates, false_negative_rates = calculate_error_rates(
        genuine_similarities,
        imposter_similarities,
    )

    x = np.linspace(0, 1, num=100)

    output_path = output_dir / f"{model_type}_far_frr_curve.png"
    plt.figure(figsize=(8, 8))
    plt.plot(x, false_positive_rates, linewidth=2, label="FAR", color="red")
    plt.plot(x, false_negative_rates, linewidth=2, label="FRR", color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title(f"{model_type.title()} FAR-FRR curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def calculate_error_rates(
    genuine_similarities: np.ndarray,
    imposter_similarities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate false accept and false reject rates for fixed thresholds.

    Args:
        genuine_similarities: Same-speaker similarity array.
        imposter_similarities: Genuine-vs-imposter similarity array.

    Returns:
        Thresholds, false accept rates, and false reject rates.

    Raises:
        ValueError: If either similarity array is empty.
    """
    if genuine_similarities.size == 0:
        raise ValueError("genuine_similarities must contain at least one value.")
    if imposter_similarities.size == 0:
        raise ValueError("imposter_similarities must contain at least one value.")

    thresholds = np.linspace(0, 1, num=100)

    false_accept_rates: list[float] = []
    false_reject_rates: list[float] = []
    for threshold in thresholds:
        false_accept_rate = float(np.mean(imposter_similarities >= threshold))
        false_reject_rate = float(np.mean(genuine_similarities < threshold))
        false_accept_rates.append(false_accept_rate)
        false_reject_rates.append(false_reject_rate)

    return (
        thresholds,
        np.asarray(false_accept_rates, dtype=np.float64),
        np.asarray(false_reject_rates, dtype=np.float64),
    )


def find_equal_error_rate(
    false_positive_rates: np.ndarray,
    false_negative_rates: np.ndarray,
) -> tuple[int, float]:
    """
    Find the threshold index closest to the equal error rate point.

    Args:
        false_positive_rates: False positive or false accept rates.
        false_negative_rates: False negative or false reject rates.

    Returns:
        Index of the closest operating point and its averaged equal error rate.
    """
    eer_index = int(np.argmin(np.abs(false_positive_rates - false_negative_rates)))
    eer = float((false_positive_rates[eer_index] + false_negative_rates[eer_index]) / 2.0)
    return eer_index, eer


def main() -> None:
    """Run speaker selection, batched inference, similarity computation, and output writing."""
    torch.cuda.empty_cache()
    args = parse_args()
    device = torch.device(args.device)
    model_type: ModelType = args.model

    user_wavs = get_user_wav_paths(args.wav_dir)
    genuine_users, imposter_users = select_users(
        user_wavs=user_wavs,
        genuine_count=args.genuine_users,
        imposter_count=args.imposter_users,
        seed=args.seed,
    )
    genuine_paths, genuine_labels = flatten_selected_paths(user_wavs, genuine_users)
    imposter_paths, _ = flatten_selected_paths(user_wavs, imposter_users)

    model = load_model(model_type, device=device)
    genuine_embeddings = extract_embeddings(
        model=model,
        model_type=model_type,
        wav_paths=genuine_paths,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        device=device,
    )
    imposter_embeddings = extract_embeddings(
        model=model,
        model_type=model_type,
        wav_paths=imposter_paths,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        device=device,
    )

    genuine_similarities, imposter_similarities = compute_similarity_arrays(
        genuine_embeddings=genuine_embeddings,
        genuine_labels=genuine_labels,
        imposter_embeddings=imposter_embeddings,
    )

    arrays_path = save_similarity_arrays(
        output_dir=args.output_dir,
        model_type=model_type,
        genuine_similarities=genuine_similarities,
        imposter_similarities=imposter_similarities,
        genuine_users=genuine_users,
        imposter_users=imposter_users,
    )
    print(f"Saved arrays: {arrays_path}")
    print(f"Genuine similarities: {genuine_similarities.shape[0]}")
    print(f"Imposter similarities: {imposter_similarities.shape[0]}")

    if not args.no_plot:
        histogram_path = plot_similarity_histogram(
            output_dir=args.output_dir,
            model_type=model_type,
            genuine_similarities=genuine_similarities,
            imposter_similarities=imposter_similarities,
        )
        roc_path = plot_ROC_curve(
            output_dir=args.output_dir,
            model_type=model_type,
            genuine_similarities=genuine_similarities,
            imposter_similarities=imposter_similarities,
        )
        det_path = plot_DET_curve(
            output_dir=args.output_dir,
            model_type=model_type,
            genuine_similarities=genuine_similarities,
            imposter_similarities=imposter_similarities,
        )
        far_frr_path = plot_FAR_FRR_curve(
            output_dir=args.output_dir,
            model_type=model_type,
            genuine_similarities=genuine_similarities,
            imposter_similarities=imposter_similarities,
        )
        print(f"Saved histogram: {histogram_path}")
        print(f"Saved ROC curve: {roc_path}")
        print(f"Saved DET curve: {det_path}")
        print(f"Saved FAR-FRR curve: {far_frr_path}")


if __name__ == "__main__":
    main()
