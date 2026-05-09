import argparse
import queue
from typing import Any
import os
import sys

PROJECT_PATH = "/Users/masonbarlow/PycharmProjects/biometrics_final_project"
if str(PROJECT_PATH) not in sys.path:
    sys.path.append(PROJECT_PATH)

if os.getcwd() != PROJECT_PATH:
    os.chdir(PROJECT_PATH)


import numpy as np
import torch

from src.app.identify import identify, set_database_path, set_model_type
from src.app.inference import get_features
from src.app.load import load_model
from src.app.utils import audio_chunk_to_tensor


DEFAULT_SAMPLE_RATE = 44100
DESIRED_SAMPLE_RATE = 16000
DEFAULT_BLOCK_SECONDS = 4
DEFAULT_DATABASE_PATH = "data/speaker_embeddings.db"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the audio recognition inference app.

    Returns:
        Parsed command-line arguments containing model selection, audio, and device settings.
    """
    parser = argparse.ArgumentParser(description="Run speaker recognition inference on microphone audio.")
    parser.add_argument(
        "--model-type",
        choices=("homegrown", "finetuned"),
        required=True,
        help="Speaker recognition model to load for inference.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=(DEFAULT_SAMPLE_RATE,),
        default=DEFAULT_SAMPLE_RATE,
        help="Microphone capture sample rate. Audio is resampled to 16000 Hz before model inference.",
    )
    parser.add_argument(
        "--block-seconds",
        type=int,
        default=DEFAULT_BLOCK_SECONDS,
        help="Number of seconds of audio to collect before each inference pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for model inference.",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Start enrollment mode instead of the continuous inference loop.",
    )
    parser.add_argument(
        "--database-path",
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database path used to store enrolled speaker embeddings.",
    )
    return parser.parse_args()


def make_audio_callback(audio_queue: queue.Queue[np.ndarray]) -> Any:
    """
    Create a sounddevice callback that stores captured audio chunks in a queue.

    Args:
        audio_queue: Queue receiving copied microphone input chunks.

    Returns:
        Callback function compatible with sounddevice.InputStream.
    """
    def callback(indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """
        Copy each microphone input block into the inference queue.

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


def run_inference_loop(args: argparse.Namespace) -> None:
    """
    Capture microphone audio continuously and run embedding extraction on each queued chunk.

    Args:
        args: Parsed application arguments specifying model, audio, and device options.
    """
    import sounddevice as sd

    device = torch.device(args.device)
    model = load_model(args.model_type, device=device)
    set_database_path(args.database_path)
    set_model_type(args.model_type)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    sample_rate = DEFAULT_SAMPLE_RATE
    blocksize = sample_rate * args.block_seconds

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=make_audio_callback(audio_queue),
    )

    with stream:
        print(f"Listening with {args.model_type} model on {device}. Press Ctrl+C to stop.")
        while True:
            chunk = audio_queue.get()
            inference_tensor = audio_chunk_to_tensor(chunk, sample_rate, device=device)
            features = get_features(model, args.model_type, inference_tensor, sample_rate=DESIRED_SAMPLE_RATE)
            print(f"Extracted features with shape {tuple(features.shape)}")
            identify(features)


def main() -> None:
    """
    Start the command-line audio recognition inference application.
    """
    args = parse_args()
    if args.enroll:
        from src.app.enrollment import run_enrollment_loop

        run_enrollment_loop(args)
    else:
        run_inference_loop(args)


if __name__ == "__main__":
    main()
