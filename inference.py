"""BigVGAN vocoder inference module for super-resolution audio synthesis.

This module provides a high-level interface for using the BigVGAN neural vocoder
for audio super-resolution tasks. It supports chunk-based processing with
crossfade for handling long audio files efficiently.

Example:
    >>> vocoder = BigVGANVocoder.from_pretrained("nvidia/bigvgan_24khz_100band")
    >>> vocoder = vocoder.to("cuda")
    >>> result = vocoder.synthesize(audio, sampling_rate=16000)
    >>> output_audio = result["audio"]
"""

from typing import Optional, Union

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm

from bigvgan import BigVGAN
from meldataset import mel_spectrogram


class BigVGANVocoder(BigVGAN):
    """BigVGAN-based vocoder for audio super-resolution synthesis.

    This class extends BigVGAN to provide convenient methods for loading
    audio, preprocessing, and synthesizing high-resolution output with
    support for chunk processing and crossfade blending.

    The vocoder can handle variable-length inputs by processing them in
    chunks with overlap, using linear crossfade to produce seamless output.

    Attributes:
        h: Hyperparameters AttrDict containing model configuration.

    Example:
        >>> vocoder = BigVGANVocoder.from_pretrained("nvidia/bigvgan_24khz_100band")
        >>> vocoder = vocoder.to("cuda")
        >>> inputs = vocoder.load(audio="input.wav")
        >>> output = vocoder(**inputs)
    """

    def __init__(self, *args, **kwargs):
        """Initialize BigVGANVocoder.

        Args:
            *args: Positional arguments passed to BigVGAN parent class.
            **kwargs: Keyword arguments passed to BigVGAN parent class.
        """
        super().__init__(*args, **kwargs)

    def load(
        self,
        audio: Optional[Union[str, np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        reference_audio: Optional[Union[str, np.ndarray]] = None,
        reference_sampling_rate: Optional[int] = None,
    ):
        """
        Load and preprocess audio for vocoder input.

        Converts input audio to mel spectrograms at both target and degraded
        sampling rates for super-resolution synthesis.

        Args:
            audio: Input audio as file path (str) or waveform (np.ndarray).
            sampling_rate: Sampling rate of the input audio. Required if audio
                is np.ndarray, ignored if audio is a file path.
            reference_audio: Optional reference audio for conditioning. If None,
                uses the input audio as reference.
            reference_sampling_rate: Sampling rate of the reference audio.

        Returns:
            dict: Contains 'x' (degraded mel spectrogram) and 'reference_x'
                (reference mel spectrogram) as torch tensors on device.
        """
        device = next(self.parameters()).device

        if isinstance(audio, str):
            audio, sampling_rate = librosa.load(
                audio, sr=self.h.sampling_rate, mono=True
            )

        if sampling_rate != self.h.sampling_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=self.h.sampling_rate,
                res_type="scipy",
            )

        if reference_audio is None:
            reference_audio = audio
        else:
            if reference_sampling_rate != self.h.sampling_rate:
                reference_audio = librosa.resample(
                    reference_audio,
                    orig_sr=reference_sampling_rate,
                    target_sr=self.h.sampling_rate,
                    res_type="scipy",
                )

        degraded_audio = librosa.resample(
            audio,
            orig_sr=self.h.sampling_rate,
            target_sr=self.h.degraded_sampling_rate,
            res_type="scipy",
        )
        reference_audio = torch.from_numpy(reference_audio).float().view(1, -1)
        reference_mel = mel_spectrogram(
            reference_audio,
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax,
            center=False,
        ).to(device)

        degraded_audio = torch.from_numpy(degraded_audio).float().view(1, -1)
        degraded_mel = mel_spectrogram(
            degraded_audio,
            self.h.degraded_n_fft,
            self.h.num_mels,
            self.h.degraded_sampling_rate,
            self.h.degraded_hop_size,
            self.h.degraded_win_size,
            self.h.degraded_fmin,
            self.h.degraded_fmax,
            center=False,
        ).to(device)
        return {"x": degraded_mel, "reference_x": reference_mel}

    @torch.inference_mode()
    def synthesize(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        speed: float = 1.0,
        chunk_samples: int = 40960,
        overlap_samples: int = 4096,
        show_progress: bool = False,
        reference_audio: Optional[np.ndarray] = None,
        reference_sampling_rate: Optional[int] = None,
    ):
        """
        Synthesize high-resolution audio using chunk processing with crossfade.

        Args:
            audio: Input audio waveform as numpy array.
            sampling_rate: Sampling rate of the input audio.
            speed: Playback speed factor. Defaults to 1.0.
            chunk_samples: Number of INPUT samples per processing chunk. This should
                match the training segment_size. Defaults to 40960 samples
                (~1.28 seconds at 32kHz).
            overlap_samples: Number of INPUT samples to overlap between chunks.
                Defaults to 4096 samples (~0.128 seconds at 32kHz).
            show_progress: Whether to display a progress bar. Defaults to False.
            reference_audio: Optional reference audio for conditioning.
            reference_sampling_rate: Sampling rate of the reference audio.

        Returns:
            dict: Contains 'audio' (np.ndarray) and 'sampling_rate' (int).
        """
        if sampling_rate != self.h.sampling_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=self.h.sampling_rate,
                res_type="scipy",
            )
            sampling_rate = self.h.sampling_rate

        total_input_samples = len(audio)

        # If audio is short enough, process in one pass
        if total_input_samples <= chunk_samples:
            return self._synthesize_chunk(
                audio, sampling_rate, speed, reference_audio, reference_sampling_rate
            )

        # Calculate step size in INPUT samples (non-overlapping portion)
        input_step = chunk_samples - overlap_samples
        assert input_step > 0, "overlap_samples must be less than chunk_samples"

        # Calculate output overlap for crossfade (proportional to speed)
        output_overlap = int(overlap_samples / speed)

        # Pre-create fade windows (reusable)
        if output_overlap > 0:
            fade_out = np.linspace(1, 0, output_overlap, dtype=np.float32)
            fade_in = np.linspace(0, 1, output_overlap, dtype=np.float32)

        # Collect all chunks first
        chunks = []
        chunk_positions = list(range(0, total_input_samples, input_step))

        if show_progress:
            iterator = tqdm(chunk_positions, total=len(chunk_positions))
        else:
            iterator = chunk_positions

        for input_start in iterator:
            input_end = min(input_start + chunk_samples, total_input_samples)
            audio_chunk = audio[input_start:input_end]

            # Zero-pad if chunk is shorter than expected
            if len(audio_chunk) < chunk_samples:
                audio_chunk = np.pad(
                    audio_chunk, (0, chunk_samples - len(audio_chunk)), mode="constant"
                )

            # Synthesize chunk
            chunk_output = self._synthesize_chunk(
                audio_chunk,
                sampling_rate,
                speed,
                reference_audio,
                reference_sampling_rate,
            )["audio"]

            # Trim padding effect from output (only for last chunk)
            if input_end < input_start + chunk_samples:
                valid_input_len = input_end - input_start
                valid_output_len = int(valid_input_len / speed)
                chunk_output = chunk_output[:valid_output_len]

            chunks.append(chunk_output)

        # Concatenate chunks with crossfade (optimized numpy operations)
        if len(chunks) == 1:
            result = chunks[0]
        else:
            # Estimate total output length for pre-allocation
            total_len = sum(len(c) for c in chunks) - output_overlap * (len(chunks) - 1)
            result = np.zeros(total_len, dtype=np.float32)

            # Copy first chunk
            write_pos = len(chunks[0])
            result[:write_pos] = chunks[0]

            for i in range(1, len(chunks)):
                curr_chunk = chunks[i]
                actual_overlap = min(output_overlap, write_pos, len(curr_chunk))

                if actual_overlap > 0:
                    # Apply crossfade in-place
                    overlap_start = write_pos - actual_overlap
                    result[overlap_start:write_pos] *= fade_out[:actual_overlap]
                    result[overlap_start:write_pos] += (
                        curr_chunk[:actual_overlap] * fade_in[:actual_overlap]
                    )

                    # Copy non-overlapping portion
                    non_overlap_len = len(curr_chunk) - actual_overlap
                    if non_overlap_len > 0:
                        end_pos = write_pos + non_overlap_len
                        if end_pos <= len(result):
                            result[write_pos:end_pos] = curr_chunk[actual_overlap:]
                        else:
                            # Extend result if needed (edge case)
                            result = np.concatenate(
                                [result[:write_pos], curr_chunk[actual_overlap:]]
                            )
                            end_pos = len(result)
                        write_pos = end_pos
                else:
                    # No overlap, just append
                    end_pos = write_pos + len(curr_chunk)
                    if end_pos <= len(result):
                        result[write_pos:end_pos] = curr_chunk
                    else:
                        result = np.concatenate([result[:write_pos], curr_chunk])
                        end_pos = len(result)
                    write_pos = end_pos

            # Trim to actual length
            result = result[:write_pos]

        return {"audio": result, "sampling_rate": self.h.sampling_rate}

    def _synthesize_chunk(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        speed: float,
        reference_audio: Optional[np.ndarray],
        reference_sampling_rate: Optional[int],
    ):
        """
        Synthesize a single chunk of audio.

        Args:
            audio: Input audio chunk as numpy array.
            sampling_rate: Sampling rate of the input audio.
            speed: Playback speed factor.
            reference_audio: Optional reference audio for conditioning.
            reference_sampling_rate: Sampling rate of the reference audio.

        Returns:
            dict: Contains 'audio' (np.ndarray) and 'sampling_rate' (int).
        """
        inputs = self.load(
            audio=audio,
            sampling_rate=sampling_rate,
            reference_audio=reference_audio,
            reference_sampling_rate=reference_sampling_rate,
        )
        generated_audio = self(speed=speed, **inputs)
        generated_audio = generated_audio.view(-1).detach().cpu().numpy()

        # Explicitly release GPU tensors
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"audio": generated_audio, "sampling_rate": self.h.sampling_rate}
