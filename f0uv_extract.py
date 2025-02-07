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


import json

file_path = "/home/junchuan/EnCodec_Trainer/LibriTTS_meta.json"

f0_root = "/data2/junchuan/libriTTS_prosody/f0"
uv_root = "/data2/junchuan/libriTTS_prosody/uv"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for wav_path in data:
    waveform, sample_rate = torchaudio.load(wav_path)

    # All resample to 24000Hz
    transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
    if sample_rate != 24000:
        waveform = transform(waveform)

    f0, uv = extractor_pyworld(waveform, 24000, 40/24000 * 1000)

    f0_tensor = torch.tensor(f0, device=waveform.device).unsqueeze(0)
    uv_tensor = torch.tensor(uv, device=waveform.device).unsqueeze(0)

    file_name = wav_path.split('/')[-1]
    new_file_name = file_name.replace('.wav', '.pt')

    torch.save(f0_tensor, os.path.join(f0_root, new_file_name))
    torch.save(uv_tensor, os.path.join(uv_root, new_file_name))
