"""Utility functions for BigVGAN training and inference.

This module provides helper functions for checkpoint management, weight
initialization, visualization, and audio I/O operations.

Adapted from https://github.com/jik876/hifi-gan under the MIT license.
LICENSE is in incl_licenses directory.
"""

import logging
import glob
import os

import matplotlib
import torch
from torch.nn.utils import weight_norm

matplotlib.use("Agg")
import matplotlib.pylab as plt
import soundfile as sf


def plot_spectrogram_to_numpy(spectrogram):
    """Convert spectrogram to numpy RGB array for logging.

    Args:
        spectrogram: 2D array-like spectrogram to visualize.

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3).
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_spectrogram(spectrogram):
    """Create matplotlib figure of spectrogram visualization.

    Args:
        spectrogram: 2D array-like spectrogram to visualize.

    Returns:
        matplotlib.figure.Figure: Figure object containing the spectrogram plot.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_spectrogram_clipped(spectrogram, clip_max=2.0):
    """Create matplotlib figure of spectrogram with clipped values.

    Args:
        spectrogram: 2D array-like spectrogram to visualize.
        clip_max: Maximum value for colormap clipping. Defaults to 2.0.

    Returns:
        matplotlib.figure.Figure: Figure object containing the clipped spectrogram plot.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=1e-6,
        vmax=clip_max,
    )
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    """Initialize convolutional layer weights with normal distribution.

    Args:
        m: PyTorch module to initialize.
        mean: Mean of the normal distribution. Defaults to 0.0.
        std: Standard deviation of the normal distribution. Defaults to 0.01.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    """Apply weight normalization to convolutional layers.

    Args:
        m: PyTorch module to apply weight normalization to.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    """Calculate padding size for same-size convolution output.

    Args:
        kernel_size: Size of the convolutional kernel.
        dilation: Dilation rate of the convolution. Defaults to 1.

    Returns:
        int: Padding size to maintain input dimensions.
    """
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    """Load model checkpoint from file.

    Args:
        filepath: Path to the checkpoint file.
        device: Device to load the checkpoint onto (e.g., 'cpu', 'cuda').

    Returns:
        dict: Loaded checkpoint dictionary containing model state and metadata.

    Raises:
        AssertionError: If the specified file does not exist.
    """
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    """Save model checkpoint to file.

    Args:
        filepath: Path where the checkpoint will be saved.
        obj: Dictionary containing model state and metadata to save.
    """
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix, renamed_file=None):
    """Scan directory for the latest checkpoint file.

    Searches for checkpoint files matching the pattern '{prefix}????????'
    and returns the most recent one. Falls back to renamed_file if no
    pattern-based checkpoints are found.

    Args:
        cp_dir: Directory to scan for checkpoints.
        prefix: Prefix pattern for checkpoint files (e.g., 'g_' or 'do_').
        renamed_file: Optional fallback filename to check if no pattern
            matches are found. Defaults to None.

    Returns:
        Optional[str]: Path to the latest checkpoint file, or None if not found.
    """
    # Fallback to original scanning logic first
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)

    if len(cp_list) > 0:
        last_checkpoint_path = sorted(cp_list)[-1]
        print(f"[INFO] Resuming from checkpoint: '{last_checkpoint_path}'")
        return last_checkpoint_path

    # If no pattern-based checkpoints are found, check for renamed file
    if renamed_file:
        renamed_path = os.path.join(cp_dir, renamed_file)
        if os.path.isfile(renamed_path):
            print(f"[INFO] Resuming from renamed checkpoint: '{renamed_file}'")
            return renamed_path

    return None


def save_audio(audio, path, sr):
    """Save audio waveform to file.

    Args:
        audio: Audio waveform as numpy array.
        path: Output file path.
        sr: Sampling rate of the audio.
    """
    sf.write(path, audio, sr)
