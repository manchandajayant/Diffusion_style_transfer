import argparse
import os
import pickle
import warnings

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

TARGET_SR = 44100
CLIP_DURATION_SECONDS = 5.0
TARGET_SAMPLES = int(TARGET_SR * CLIP_DURATION_SECONDS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_view_audio(out_file, sr=44100):
    # Load saved audio
    sr = int(sr)
    if sr > 65535:
        sr = 44100
    audio_data, sr = sf.read(out_file)

    # If stereo, convert to mono for plotting
    if audio_data.ndim > 1:
        audio_mono = librosa.to_mono(audio_data.T)
    else:
        audio_mono = audio_data

    # Plot waveform
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(audio_mono, sr=sr)
    plt.title(f"Waveform of {out_file}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # Play audio (in notebook)
    # ipd.display(ipd.Audio(audio_data, rate=44100))
    ipd.display(ipd.Audio(out_file))


# Helpers
def load_audio_to_numpy(path, sr=44100):
    wav, fs = sf.read(path)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=1)
    if fs != sr:
        import librosa

        wav = librosa.resample(wav.T, fs, sr).T
    L = wav.shape[0]
    if L < TARGET_SAMPLES:
        wav = np.pad(wav, ((0, TARGET_SAMPLES - L), (0, 0)))
    else:
        wav = wav[:TARGET_SAMPLES]
    return wav.astype(np.float32)


def load_audio(path):
    arr = load_audio_to_numpy(path)
    return torch.from_numpy(arr).permute(1, 0).unsqueeze(0).to(device)


def get_audio_files(folder):
    exts = (".wav", ".flac", ".mp3")
    return [
        os.path.join(r, f)
        for r, _, fs in os.walk(folder)
        for f in fs
        if f.lower().endswith(exts)
    ]
