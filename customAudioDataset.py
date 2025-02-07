import os
import pandas as pd
import torch
import torchaudio
import random
import json
import torchaudio.transforms as T


# import pesto
import multiprocessing
import soundfile as sf
import pyworld as pw
# You can also predict pitches from audio files directly
import os
import numpy as np
import math
from scipy.signal import resample_poly
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

def normalize_f0(f0):
    # check all 0 case
    if np.all(f0 == 0):
        return np.zeros_like(f0, dtype=int)
    
    non_zero_f0 = f0[f0 != 0]
    
    # normalize to 0-1
    if non_zero_f0.size > 0:
        f0_min = np.min(non_zero_f0)
        f0_max = np.max(non_zero_f0)
    
        if f0_max == f0_min:
            normalized_non_zero_f0 = np.zeros_like(non_zero_f0)
        else:
            normalized_non_zero_f0 = (non_zero_f0 - f0_min) / (f0_max - f0_min)
    else:
        normalized_non_zero_f0 = np.zeros_like(non_zero_f0)

    quantized_f0 = np.clip(np.round(normalized_non_zero_f0 * 254) + 1, 1, 255).astype(int)

    result_f0 = np.zeros_like(f0, dtype=int)
    result_f0[f0 != 0] = quantized_f0

    return result_f0

def extractor_pyworld(x, sr, frame_shift):
    if isinstance(x, torch.Tensor):
        x = x.numpy().flatten()
    x = x.astype(np.float64)

    f0, _ = pw.harvest(x, sr, frame_period=frame_shift)
    f0 = normalize_f0(f0)
    uv = (f0 > 0).astype(np.float32)
    return f0[:-1], uv[:-1]

# def pitch_to_f0(pitch):
#     if pitch == 0:
#         return 0
#     return 27.5 * math.pow(2, (pitch - 21) / 12)
    
# def f0ToPitch(f0):
#     return np.log2(f0 / 27.5) * 12 + 21

# def coarse_f0(f0, f0_bin):
#     f0_mel = 1127 * np.log(1 + f0 / 700)
#     f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (
#         f0_bin - 2
#     ) / (f0_mel_max - f0_mel_min) + 1

#     # use 0 or 1
#     f0_mel[f0_mel <= 1] = 1
#     f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
#     f0_coarse = np.rint(f0_mel).astype(int)
#     assert f0_coarse.max() <= (f0_bin - 1) and f0_coarse.min() >= 1, (
#         f0_coarse.max(),
#         f0_coarse.min(),
#     )
#     return f0_coarse

f0_floor = 60
f0_ceil = 1400
f0_mel_min = 1127 * np.log(1 + f0_floor / 700)
f0_mel_max = 1127 * np.log(1 + f0_ceil / 700)



class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform=None, tensor_cut=0, fixed_length=None):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.meta_data = meta_data
        self.transform = transform
        self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return len(self.meta_data)

    def __getitem__(self, idx):
        audio_path = self.meta_data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        # All resample to 24000Hz
        self.transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
        if sample_rate != 24000:
            waveform = self.transform(waveform)
        # try:
        #     f0, uv = extractor_pyworld(waveform, 24000, 40/24000 * 1000)
        # except Exception as e:
        #     print(waveform.min(), waveform.max())
        #     print(f0.min(), f0.max())
        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
                waveform = waveform[:, start:start+self.tensor_cut]
            else:
                pad_size = self.tensor_cut - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_size))
        f0, uv = extractor_pyworld(waveform, 24000, 40/24000 * 1000)

        f0_tensor = torch.tensor(f0, device=waveform.device).unsqueeze(0)
        uv_tensor = torch.tensor(uv, device=waveform.device).unsqueeze(0)
            
        return waveform, sample_rate, f0_tensor, uv_tensor
