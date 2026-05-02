import queue
import select
import sys
import termios
import tty
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch import nn

from src.app.database import connect_database, initialize_database, insert_user_embedding
from src.app.inference import ModelType, get_features
from src.app.load import load_model


DEFAULT_SAMPLE_RATE = 44100
DESIRED_SAMPLE_RATE = 16000
ENROLLMENT_CLIP_SECONDS = 4
ENROLLMENT_RECORDINGS = 8


def run_enrollment_loop(args: Namespace) -> None:
    """
    Enroll a new speaker by recording multiple utterances and storing their embeddings.

    Args:
        args: Parsed command-line arguments with model, device, sample rate, and database settings.
    """
    name = prompt_for_name()
    target = prompt_for_target_status()
    device = torch.device(args.device)
    model = load_model(args.model_type, device=device)

    print(f"Enrolling {name}. Press the space bar before each {ENROLLMENT_CLIP_SECONDS}-second recording.")
    for recording_number in range(1, ENROLLMENT_RECORDINGS + 1):
        wait_for_spacebar(recording_number, ENROLLMENT_RECORDINGS)
        audio_chunk = record_audio_clip(
            sample_rate=DEFAULT_SAMPLE_RATE,
            clip_seconds=ENROLLMENT_CLIP_SECONDS,
        )
        embedding = extract_enrollment_embedding(
            model=model,
            model_type=args.model_type,
            audio_chunk=audio_chunk,
            sample_rate=DEFAULT_SAMPLE_RATE,
            device=device,
        )
        user_embedding_id = put_embedding_in_database(
            database_path=args.database_path,
            name=name,
            embedding=embedding,
            model_type=args.model_type,
            target=target,
        )
        print(f"Saved recording {recording_number}/{ENROLLMENT_RECORDINGS} as embedding {user_embedding_id}.")

    print(f"Enrollment complete for {name}.")


def prompt_for_name() -> str:
    """
    Prompt for the speaker name used in the biometric database.

    Returns:
        Non-empty speaker name entered by the user.
    """
    while True:
        name = input("Enter the user's name: ").strip()
        if name:
            return name
        print("Name cannot be empty.")


def prompt_for_target_status() -> bool:
    """
    Prompt for whether the enrolled speaker should be marked as a target.

    Returns:
        True when the user answers yes, and False when the user answers no.
    """
    while True:
        answer = input("Is this user a target? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def wait_for_spacebar(recording_number: int, total_recordings: int) -> None:
    """
    Block until the user presses the space bar to begin the next enrollment clip.

    Args:
        recording_number: One-based index of the next recording.
        total_recordings: Total number of recordings required for enrollment.
    """
    print(f"Press space to start recording {recording_number}/{total_recordings}.")
    if not sys.stdin.isatty():
        input("Press Enter to start recording.")
        return

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [])
            if readable and sys.stdin.read(1) == " ":
                return
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)


def make_audio_callback(audio_queue: queue.Queue[np.ndarray]) -> Any:
    """
    Create a sounddevice callback that stores one-shot enrollment audio blocks.

    Args:
        audio_queue: Queue receiving copied microphone input blocks.

    Returns:
        Callback function compatible with sounddevice.InputStream.
    """
    def callback(indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """
        Copy each microphone input block into the enrollment queue.

        Args:
            indata: Captured audio block with shape [frames, channels].
            frames: Number of frames in the captured audio block.
            time: Timing metadata provided by sounddevice.
            status: Input stream status flags.
        """
        if status:
            print(status)
        audio_queue.put(indata.copy())

    return callback


def record_audio_clip(sample_rate: int, clip_seconds: int) -> np.ndarray:
    """
    Record one microphone clip using the same InputStream queue pattern as inference.

    Args:
        sample_rate: Microphone sample rate for capture.
        clip_seconds: Number of seconds to capture in a single recording.

    Returns:
        Recorded audio block shaped [samples, channels].
    """
    import sounddevice as sd

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    blocksize = sample_rate * clip_seconds
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=make_audio_callback(audio_queue),
    )

    print("Recording...")
    with stream:
        audio_chunk = audio_queue.get()
    print("Recording finished.")
    return audio_chunk


def audio_chunk_to_tensor(chunk: np.ndarray, sample_rate: int, device: torch.device) -> torch.Tensor:
    """
    Convert a recorded enrollment clip into a 16 kHz mono float tensor.

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


def extract_enrollment_embedding(
    model: nn.Module,
    model_type: ModelType,
    audio_chunk: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> list[float]:
    """
    Convert a recorded audio clip into a speaker embedding ready for database storage.

    Args:
        model: Loaded speaker recognition model.
        model_type: Model family that should extract the embedding.
        audio_chunk: Recorded microphone audio block.
        sample_rate: Sampling rate of the recorded audio.
        device: Torch device used for embedding extraction.

    Returns:
        Speaker embedding as a flat list of float values.
    """
    inference_tensor = audio_chunk_to_tensor(audio_chunk, sample_rate, device=device)
    features = get_features(model, model_type, inference_tensor, sample_rate=DESIRED_SAMPLE_RATE)
    return features.detach().cpu().reshape(-1).tolist()


def put_embedding_in_database(
    database_path: str | Path,
    name: str,
    embedding: list[float],
    model_type: ModelType,
    target: bool,
) -> int:
    """
    Store one speaker embedding in the biometric database for the current user.

    Args:
        database_path: SQLite database path for enrolled speaker embeddings.
        name: User name associated with the embedding.
        embedding: Speaker embedding vector to store.
        model_type: Model family that produced the embedding.
        target: Whether a newly inserted user should be marked as a target.

    Returns:
        New user_embeddings row ID linking the user to the stored vector.
    """
    database_file = Path(database_path)
    database_exists = database_file.exists()
    database_file.parent.mkdir(parents=True, exist_ok=True)

    connection = connect_database(database_file)
    try:
        if not database_exists:
            initialize_database(connection)
        return insert_user_embedding(
            connection=connection,
            name=name,
            embedding=embedding,
            model_type=model_type,
            target=target,
        )
    finally:
        connection.close()
