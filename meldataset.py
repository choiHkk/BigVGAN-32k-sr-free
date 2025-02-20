# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import io
import math
import os
import pathlib
import random
from typing import List, Optional, Tuple

import datasets
import librosa
import numpy as np
import torch
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from pydub import AudioSegment
from tqdm import tqdm

from env import AttrDict

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


class Collator:
    def __init__(self, h, split=True):
        self.h = h
        self.split = split
        self.use_reference_encoder = self.h.get("use_reference_encoder", False)

    def __call__(self, samples):
        mel_batch = []
        audio_batch = []
        mel_loss_batch = []
        reference_mel_batch = []

        for item in samples:
            try:
                audio, source_sampling_rate = load_audio_from_bytes(item["audio"])
                if source_sampling_rate > self.h.sampling_rate:
                    audio = librosa.resample(
                        audio,
                        orig_sr=source_sampling_rate,
                        target_sr=self.h.sampling_rate,
                        res_type="scipy",
                    )
                    source_sampling_rate = self.h.sampling_rate

            except:
                audio = 2 * np.random.rand(self.h.sampling_rate) - 1
                source_sampling_rate = self.h.sampling_rate

            if source_sampling_rate != self.h.sampling_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=source_sampling_rate,
                    target_sr=self.h.sampling_rate,
                    res_type="scipy",
                )

            if self.split:
                # Compute upper bound index for the random chunk
                random_chunk_upper_bound = max(0, audio.shape[0] - self.h.segment_size)

                # Crop or pad audio to obtain random chunk with target_segment_size
                if audio.shape[0] >= self.h.segment_size:
                    audio_start = random.randint(0, random_chunk_upper_bound)
                    audio = audio[audio_start : audio_start + self.h.segment_size]
                else:
                    audio = np.pad(
                        audio,
                        (0, self.h.segment_size - audio.shape[0]),
                        mode="constant",
                    )

                # Compute upper bound index for the random chunk
                random_chunk_upper_bound = max(0, audio.shape[0] - self.h.segment_size)

                # Crop or pad audio to obtain random chunk with target_segment_size
                if audio.shape[0] >= self.h.segment_size:
                    audio_start = random.randint(0, random_chunk_upper_bound)
                    reference_audio = audio[
                        audio_start : audio_start + self.h.segment_size
                    ]
                else:
                    reference_audio = np.pad(
                        audio,
                        (0, self.h.segment_size - audio.shape[0]),
                        mode="constant",
                    )

                degraded_audio = librosa.resample(
                    audio,
                    orig_sr=self.h.sampling_rate,
                    target_sr=self.h.degraded_sampling_rate,
                    res_type="scipy",
                )

                if degraded_audio.shape[0] < self.h.degraded_segment_size:
                    degraded_audio = np.pad(
                        degraded_audio,
                        (0, self.h.degraded_segment_size - degraded_audio.shape[0]),
                        mode="constant",
                    )

            else:  # Validation step
                degraded_audio = librosa.resample(
                    audio,
                    orig_sr=self.h.sampling_rate,
                    target_sr=self.h.degraded_sampling_rate,
                    res_type="scipy",
                )

                min_frame_len = min(
                    audio.shape[-1] // self.h.hop_size,
                    degraded_audio.shape[-1] // self.h.degraded_hop_size,
                )

                audio_min_len = int(min_frame_len * self.h.hop_size)
                degraded_audio_min_len = int(min_frame_len * self.h.degraded_hop_size)

                audio = audio[:audio_min_len]
                reference_audio = audio
                degraded_audio = degraded_audio[:degraded_audio_min_len]

            # BigVGAN is trained using volume-normalized waveform
            audio = librosa.util.normalize(audio) * 0.95
            reference_audio = librosa.util.normalize(reference_audio) * 0.95
            degraded_audio = librosa.util.normalize(degraded_audio) * 0.95

            # Cast ndarray to torch tensor
            audio = (
                torch.from_numpy(audio).float().unsqueeze(0)
            )  # [B(1), self.segment_size]
            reference_audio = (
                torch.from_numpy(reference_audio).float().unsqueeze(0)
            )  # [B(1), self.segment_size]
            degraded_audio = (
                torch.from_numpy(degraded_audio).float().unsqueeze(0)
            )  # [B(1), self.degraded_segment_size]

            mel = mel_spectrogram(
                degraded_audio,
                self.h.degraded_n_fft,
                self.h.num_mels,
                self.h.degraded_sampling_rate,
                self.h.degraded_hop_size,
                self.h.degraded_win_size,
                self.h.degraded_fmin,
                self.h.degraded_fmax,
                center=False,
            )  # [B(1), self.num_mels, self.degraded_segment_size // self.degraded_hop_size]

            reference_mel = mel_spectrogram(
                reference_audio,
                self.h.n_fft,
                self.h.num_mels,
                self.h.sampling_rate,
                self.h.hop_size,
                self.h.win_size,
                self.h.fmin,
                self.h.fmax_loss,
                center=False,
            )  # [B(1), self.num_mels, self.segment_size // self.hop_size]

            mel_loss = mel_spectrogram(
                audio,
                self.h.n_fft,
                self.h.num_mels,
                self.h.sampling_rate,
                self.h.hop_size,
                self.h.win_size,
                self.h.fmin,
                self.h.fmax_loss,
                center=False,
            )  # [B(1), self.num_mels, self.segment_size // self.hop_size]

            deg_T = mel.shape[-1]
            ori_T = mel_loss.shape[-1]
            assert deg_T == ori_T, (
                deg_T,
                degraded_audio.shape[-1],
                ori_T,
                audio.shape[-1],
            )

            mel_batch.append(mel)
            audio_batch.append(audio)
            mel_loss_batch.append(mel_loss)
            reference_mel_batch.append(reference_mel)

        mel_batch = torch.cat(mel_batch, dim=0)
        audio_batch = torch.cat(audio_batch, dim=0)
        mel_loss_batch = torch.cat(mel_loss_batch, dim=0)
        reference_mel_batch = torch.cat(reference_mel_batch, dim=0)

        return mel_batch, reference_mel_batch, audio_batch.squeeze(1), mel_loss_batch


def load_audio_from_bytes(audio_samples):
    value = AudioSegment.from_file(io.BytesIO(audio_samples))

    if value.channels == 2:
        value = value.set_channels(1)

    audio = np.array(value.get_array_of_samples())
    sampling_rate = value.frame_rate

    audio = audio / np.iinfo(audio.dtype).max

    if audio.min() < -1 or audio.max() > 1:
        audio = librosa.util.normalize(audio)

    return audio, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


mel_basis_cache = {}
hann_window_cache = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


def get_mel_spectrogram(wav, h):
    """
    Generate mel spectrogram from a waveform using given hyperparameters.

    Args:
        wav (torch.Tensor): Input waveform.
        h: Hyperparameters object with attributes n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    return mel_spectrogram(
        wav,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )


def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ["." + x if x[0] != "." else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == "."
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any([keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words]
                    )
                    if (
                        has_ext
                        and has_keyword
                        and not has_banned
                        and not is_hidden
                        and not os.path.basename(f.path).startswith("._")
                    ):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def fast_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions,
    # max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = [
        "." + x if x[0] != "." else x for x in ext
    ]  # add starting period to extensions if needed
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=[".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus"],
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:  # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames


def get_dataset(data_patterns, rank=0, cache_dir=None):
    trainset = []
    for pattern in data_patterns:
        data_files = get_audio_filenames([pattern], exts=[".tar"])
        dataset = datasets.load_dataset(
            "webdataset",
            data_files=data_files,
            cache_dir=cache_dir,
            num_proc=16,
        )
        dataset = dataset["train"]
        dataset = dataset.remove_columns(["json"])
        trainset.append(dataset)

    trainset = datasets.concatenate_datasets(trainset)
    trainset = trainset.shuffle(seed=42 + rank)
    return trainset
