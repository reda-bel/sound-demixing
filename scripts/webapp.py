import os
import torch
import librosa
import torchaudio
import soundfile
import numpy as np
import streamlit as st
import librosa.display
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from my_submission.ensemble import EnsembleNet
from torchaudio.transforms import Fade


st.set_page_config(page_title="aimless-cdx-splitter")
st.image("docs/aimless-logo-crop.svg", use_column_width="always")

""" bundle = HDEMUCS_HIGH_MUSDB_PLUS
sample_rate = bundle.sample_rate """
fade_overlap = 0.1


@st.experimental_singleton
def load_model():
    print("Loading model...")
    model = EnsembleNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


def plot_spectrogram(y, *, sample_rate, figsize=(12, 3)):
    # Convert to mono
    if y.ndim > 1:
        y = y[0]

    fig = plt.figure(figsize=figsize)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=120)
    img = librosa.display.specshow(D, y_axis="linear", x_axis="time", sr=sample_rate)
    fig.colorbar(img, format="%+2.f dB")
    st.pyplot(fig=fig, clear_figure=True)


def process_file(file, model: torch.nn.Module, device: torch.device):
    import tempfile
    import shutil
    from pathlib import Path

    # Cache file to disk so we can read it with ffmpeg
    with tempfile.NamedTemporaryFile("wb", suffix=Path(file.name).suffix) as f:
        shutil.copyfileobj(file, f)
        # duration = librosa.get_duration(filename=f.name)
        x, sample_rate = soundfile.read(f.name)
    st.subheader("Mix")
    x_numpy = x.numpy()
    plot_spectrogram(x_numpy, sample_rate=sample_rate)
    st.audio(x_numpy, sample_rate=sample_rate)

    waveform = x.to(device)

    # split into 10.0 sec chunks
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization
    separated_music_arrays, output_sample_rates = model.separate_music_files(x, sample_rate)
    instruments = ['dialog', 'effect', 'music']

    for audio, sr in zip(separated_music_arrays.items(), output_sample_rates):
        st.subheader(audio[0].capitalize())
        plot_spectrogram(audio[1], sample_rate=sr)
        st.audio(audio[1], sample_rate=sr)


# load pretrained model
separation_model = load_model()

# load audio
uploaded_file = st.file_uploader("Choose a file to demix.")

if uploaded_file is not None:
    # split with hdemucs
    hdemucs_sources = process_file(uploaded_file, separation_model, "cuda:0")
