```python
from bigvgan import load_hparams_from_json, BigVGAN
from utils import load_checkpoint


h = load_hparams_from_json("configs/bigvgan_reference.json")
generator = BigVGAN(h)

state_dict_g = load_checkpoint("./exp/test_v4/g_01000000", "cpu")
generator.load_state_dict(state_dict_g["generator"])
generator.remove_weight_norm()
generator.save_pretrained("./pretrained_models/test")
```
```python
from inference import BigVGANVocoder

vocoder = BigVGANVocoder.from_pretrained("./pretrained_models/test")
vocoder = vocoder.cuda()
```
```python
import librosa
import numpy as np

input_path = "./test_input_16k_or_24k.wav"
reference_path = "./samples/test_output_to_be_32k.wav"

input_audio, _ = librosa.load(input_path, sr=vocoder.h.sampling_rate, mono=True)
reference_audio, _ = librosa.load(reference_path, sr=vocoder.h.sampling_rate, mono=True)

input_audio = np.clip(input_audio, a_min=-1, a_max=1)
reference_audio = np.clip(reference_audio, a_min=-1, a_max=1)
```
```python
speed = 1.0 # speaking rate (only support 1.0 ~ 2.0)

inputs = {
    "audio": input_audio, 
    "sampling_rate": vocoder.h.sampling_rate, 
    "reference_audio": reference_audio, 
    "reference_sampling_rate": vocoder.h.sampling_rate, 
    "speed": speed, 
    "show_progress": True, 
}
outputs = vocoder.synthesize2(**inputs)
```
```python
import IPython.display as ipd


display(
    ipd.Audio(
        reference_audio,  
        rate=vocoder.h.sampling_rate, 
        normalize=False
    )
)

display(
    ipd.Audio(
        outputs["audio"].T, 
        rate=outputs["sampling_rate"], 
        normalize=False
    )
)
```
