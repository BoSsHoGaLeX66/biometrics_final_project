import torch
import torchaudio
from pathlib import Path
import re
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf



def wav_to_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """
    Convert a waveform tensor into a log-Mel spectrogram.

    Args:
        waveform:
            Tensor shaped [num_samples] or [channels, num_samples].
        sample_rate:
            Sampling rate of the waveform.
        n_mels:
            Number of Mel bins.
        n_fft:
            FFT window size.
        hop_length:
            Number of samples between frames.
        win_length:
            Window size.

    Returns:
        Tensor shaped [n_mels, time_steps].
    """
    # Convert stereo to mono if needed
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    # Add batch/channel dimension expected by torchaudio transforms
    waveform = waveform.unsqueeze(0)  # [1, num_samples]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel = mel_transform(waveform)      # [1, n_mels, time]
    mel = mel.squeeze(0)              # [n_mels, time]

    # Convert to log scale
    log_mel = torch.log(mel + 1e-6)

    return log_mel


def create_audio_datasets(path_dir, sr, num_speakers, train_pct=0.8, val_pct=0.1):
    DATA_DIR = Path(path_dir)
    MAX_SEQ_LENGTH = 500
    X = []
    y = []
    pattern = re.compile('data/wav/id1(\d+)')
    for i, speaker_dir in enumerate(DATA_DIR.iterdir()):
        if i >= num_speakers:
            break

        if speaker_dir.is_dir():
            for wav_path in speaker_dir.rglob("*.wav"):
              audio, sample_rate = sf.read(wav_path)
              waveform = torch.tensor(audio, dtype=torch.float32)

              if sample_rate != sr:
                  resampler = torchaudio.transforms.Resample(sample_rate, sr)
                  waveform = resampler(waveform)

              mel = wav_to_log_mel(waveform, sr)
              mel = mel.transpose(0, 1)
              if mel.shape[0] > MAX_SEQ_LENGTH:
                  X.append(mel[:MAX_SEQ_LENGTH])
              else:
                  X.append(mel)
              m = pattern.match(str(speaker_dir))
              id_str = m.group(1)
              label = int(id_str) - 1
              y.append(label)
    X = pad_sequence(X, batch_first=True)
    X = X.permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.long)

    shuffle_idx = torch.randperm(X.size(0))

    X = X[shuffle_idx]
    y = y[shuffle_idx]

    unique_ids = torch.unique(y)

    id_map = {id.item(): i for i, id in enumerate(unique_ids)}

    y = [id_map[id.item()] for id in y]
    y = torch.tensor(y, dtype=torch.long)

    train_idx = int(len(X) * train_pct)
    val_idx = int(len(X) * val_pct)

    X_train = X[:train_idx]
    y_train = y[:train_idx]

    X_val = X[train_idx:val_idx + train_idx]
    y_val = y[train_idx:val_idx + train_idx]

    X_test = X[val_idx + train_idx:]
    y_test = y[val_idx + train_idx:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    return train_ds, val_ds, test_ds, num_speakers
