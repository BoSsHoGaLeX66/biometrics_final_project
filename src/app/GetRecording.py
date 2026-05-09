import sounddevice as sd
from scipy.io.wavfile import write
import torch


def getSoundFromMic():
    # Parameters
    fs = 44100  # Sample rate
    seconds = 4  # Duration
    filename = "output.wav"

    print("Recording...")
    # Record audio
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Done.")

    recording_tensor = torch.from_numpy(myrecording).float()

    return recording_tensor


