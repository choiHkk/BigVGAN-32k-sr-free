from typing import Optional, Union

import librosa
import numpy as np
import torch
from scipy import signal
from tqdm.auto import tqdm

from bigvgan import BigVGAN, load_hparams_from_json
from meldataset import mel_spectrogram
from utils import load_checkpoint


class BigVGANVocoder(BigVGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(
        self,
        audio: Optional[Union[str, np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        reference_audio: Optional[Union[str, np.ndarray]] = None,
        reference_sampling_rate: Optional[int] = None,
    ):
        device = next(self.parameters()).device

        if isinstance(audio, str):
            audio, sampling_rate = librosa.load(
                audio_path, sr=self.h.sampling_rate, mono=True
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
                    orig_sr=sampling_rate,
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

    def overlap_add(self, result, x, weights, start, length):
        """
        Adds the overlapping segment of the result to the result tensor.
        """
        actual_length = min(length, x.shape[-1], weights.shape[0])
        end_idx = min(start + actual_length, result.shape[-1])
        actual_length = end_idx - start

        result[..., start:end_idx] += x[..., :actual_length] * weights[:actual_length]
        return result

    @torch.inference_mode()
    def synthesize(
        self,
        audio: Optional[Union[str, np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        reference_audio: Optional[np.ndarray] = None,
        reference_sampling_rate: Optional[int] = None,
        speed: float = 1.0,
    ):
        inputs = self.load(
            audio=audio,
            sampling_rate=sampling_rate,
            reference_audio=reference_audio,
            reference_sampling_rate=reference_sampling_rate,
        )
        generated_audio = self(speed=speed, **inputs)
        generated_audio = generated_audio.view(-1).detach().cpu().numpy()
        return {"audio": generated_audio, "sampling_rate": self.h.sampling_rate}

    @torch.inference_mode()
    def synthesize2(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        speed: float = 1.0,
        overlap: int = 8,
        segment_size: int = 40960,
        show_progress: bool = False,
        reference_audio: Optional[np.ndarray] = None,
        reference_sampling_rate: Optional[int] = None,
    ):
        if sampling_rate != self.h.sampling_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=self.h.sampling_rate,
                res_type="scipy",
            )
            sampling_rate = self.h.sampling_rate

        audio = torch.from_numpy(audio).float().view(1, -1)

        chunk_size = min(self.h.hop_size * (segment_size - 1), audio.shape[1])
        window = torch.tensor(signal.windows.hamming(chunk_size), dtype=torch.float32)

        req_shape = tuple(audio.shape)
        result = torch.zeros(req_shape, dtype=torch.float32)
        counter = torch.zeros(req_shape, dtype=torch.float32)

        step = min(int(overlap * self.h.sampling_rate), chunk_size)

        if show_progress:
            total_steps = (audio.shape[1] + step - 1) // step
            iterator = tqdm(range(0, audio.shape[1], step), total=total_steps)
        else:
            iterator = range(0, audio.shape[1], step)

        for i in iterator:
            audio_chunk = audio[:, i : i + chunk_size]
            length = audio_chunk.shape[-1]
            if i + chunk_size > audio.shape[1]:
                audio_chunk = audio[:, -chunk_size:]
                length = chunk_size

            inputs = self.load(
                audio=audio_chunk.view(-1).cpu().numpy(),
                sampling_rate=sampling_rate,
                reference_audio=reference_audio,
                reference_sampling_rate=reference_sampling_rate,
            )
            x = self(speed=speed, **inputs)[0]
            x = x.cpu()

            if i + chunk_size > audio.shape[1]:
                actual_length = min(length, window.shape[0])
                end_idx = result.shape[-1]
                start_idx = end_idx - chunk_size
                counter[..., start_idx:end_idx] += window[:actual_length]

                result = self.overlap_add(
                    result, x, window, result.shape[-1] - chunk_size, actual_length
                )
            else:
                actual_length = min(length, window.shape[0])
                counter[..., i : i + actual_length] += window[:actual_length]

                result = self.overlap_add(result, x, window, i, actual_length)

        generated_audio = result / counter.clamp(min=1e-10)
        generated_audio = generated_audio.view(-1).detach().cpu().numpy()
        return {"audio": generated_audio, "sampling_rate": self.h.sampling_rate}
