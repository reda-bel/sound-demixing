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
from zftmodel import Demucs4_SeparationModel


st.set_page_config(page_title="aimless-cdx-splitter")
st.image("docs/aimless-logo-crop.svg", use_column_width="always")

fade_overlap = 0.1


@st.cache_resource
def load_model(model_label):
    print("Loading model...")
    if model_label=="Aimless":
        model = EnsembleNet()
    elif model_label=="ZFTurbo":
        model = Demucs4_SeparationModel() 
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

def process_chunk(x, sample_rate, channels, model):
    if channels==1:
        x = x[:,0]
        x_stack = np.stack([x, x], axis=1)
        separated_music_arrays, output_sample_rates = model.separate_music_file(x_stack, sample_rate)
    elif channels==2:
        x_stack = np.stack([x[:,0], x[:,1]], axis=1)
        separated_music_arrays, output_sample_rates = model.separate_music_file(x_stack, sample_rate)
    return separated_music_arrays, output_sample_rates


def process_file(file, model: torch.nn.Module, device: torch.device):
    import tempfile
    import shutil
    from pathlib import Path
    chunk_results = {'dialog':np.zeros([1, 2]),'effect':np.zeros([1, 2]),'music':np.zeros([1, 2])}
    keys = ['dialog','effect','music']
    # Cache file to disk so we can read it with ffmpeg
    with tempfile.NamedTemporaryFile("wb", suffix=Path(file.name).suffix) as f:
        shutil.copyfileobj(file, f)
        # duration = librosa.get_duration(filename=f.name)
        x, sample_rate = soundfile.read(f.name, always_2d=True)
        waveform = x
        ref = waveform.mean(0)
        
        chunks = soundfile.blocks(f.name, always_2d=True, blocksize=204800)
        channels = x.shape[1]
        for chunk in chunks:
            chunk = (chunk - ref.mean()) / ref.std()
            chunk_separated_array, sr = process_chunk(chunk, sample_rate, channels, model)
            for k in keys:
                chunk_results[k]=np.append(chunk_results[k],chunk_separated_array[k], axis=0)
    print(chunk_results)
    if channels==1:
        x = x[:,0]
        st.subheader("Mix")
        plot_spectrogram(x, sample_rate=sample_rate)
        st.audio(x, sample_rate=sample_rate)
    elif channels==2:
        st.subheader("Mix")
        plot_spectrogram(np.swapaxes(x,0,1), sample_rate=sample_rate)
        st.audio(np.swapaxes(x,0,1), sample_rate=sample_rate)

    for key in keys:
        print(chunk_results[key].shape)
        final_sep = np.swapaxes(chunk_results[key],0,1)
        st.subheader(key.capitalize())
        plot_spectrogram(final_sep,  sample_rate=sample_rate)
        st.audio(final_sep, sample_rate=sample_rate)


# load pretrained model
model_label = st.selectbox('Select Demixing Model:', options=["Aimless","ZFTurbo"])


# load audio
uploaded_file = st.file_uploader("Choose a file to demix.")


if uploaded_file is not None:
    if model_label is not None:
        separation_model = load_model(model_label)
        # split with hdemucs
        hdemucs_sources = process_file(uploaded_file, separation_model, "cuda:0")
