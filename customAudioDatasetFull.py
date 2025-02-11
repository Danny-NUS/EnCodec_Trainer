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

f0_root = "/data2/junchuan/libriTTS_prosody/f0"
uv_root = "/data2/junchuan/libriTTS_prosody/uv"

discrete_tgt = "/data2/xintong/LibriTTS_encodec_codes/train-clean-100"
# continuous_tgt = "/data2/xintong/LibriTTS_encodec_continuous"
continuous_tgt = "/data2/xintong/LibriTTS_encodec_continuous/train-clean-100"

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, target="both", transform=None, tensor_cut=0, fixed_length=None):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.meta_data = meta_data
        self.transform = transform
        self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut
        self.target = target
        if target == "discrete":
            self.tgt_path = [discrete_tgt]
        elif target == "continuous":
            self.tgt_path = [continuous_tgt]
        else:
            self.tgt_path = [discrete_tgt, continuous_tgt]

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return len(self.meta_data)

    def __getitem__(self, idx):
        audio_path = self.meta_data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        filename = os.path.splitext(os.path.basename(audio_path))[0]

        tgt_list = []
        for path_i in self.tgt_path:
            id1, id2 = filename.split("_")[:2]
            tgt_path = os.path.join(path_i, id1, id2, f"{filename}.npy")
            tgt = torch.tensor(np.load(tgt_path)).to(waveform.device)
            tgt_list.append(tgt)

        if self.transform:
            waveform = self.transform(waveform)

        # All resample to 24000Hz
        self.transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
        if sample_rate != 24000:
            waveform = self.transform(waveform)

        # Load f0 and uv
        f0_path = os.path.join(f0_root, audio_path.split("/")[-1].replace(".wav", ".pt"))
        f0 = torch.load(f0_path)
        uv = torch.logical_not((f0 > 0).float())

        tgt_list_ = []

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
                waveform = waveform[:, start:start+self.tensor_cut]
                
                start_prosody = round(start / 40)
                if start_prosody + self.tensor_cut / 40 < f0.size(1) - 1: 
                    f0 = f0[:, start_prosody:start_prosody+int(self.tensor_cut/40)]
                    uv = uv[:, start_prosody:start_prosody+int(self.tensor_cut/40)]
                else:
                    f0 = f0[:, start:]
                    uv = uv[:, start:]

                    pad_size_prosody = int(self.tensor_cut/40 - f0.size(1))
                    f0 = F.pad(f0, (0, pad_size_prosody))
                    uv = F.pad(uv, (0, pad_size_prosody))

                start_tgt = round(start / 320)

                for tgt in tgt_list:
                    if start_tgt + self.tensor_cut / 320 < tgt.size(2) - 1: 
                        tgt_list_.append(tgt[:, :, start_tgt:start_tgt + int(self.tensor_cut/320)])
                    else:
                        tgt = tgt[:, :, start:]
                        pad_size_tgt = int(self.tensor_cut/320 - tgt.size(2))
                        tgt_list_.append(F.pad(tgt, (0, pad_size_tgt)))
            else:
                pad_size = self.tensor_cut - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_size))

                pad_size_prosody = int(self.tensor_cut/40 - f0.size(1))
                f0 = F.pad(f0, (0, pad_size_prosody))
                uv = F.pad(uv, (0, pad_size_prosody))

                for tgt in tgt_list:
                    pad_size_tgt = int(self.tensor_cut/320 - tgt.size(2))
                    tgt_list_.append(F.pad(tgt, (0, pad_size_tgt)))

        # import pdb
        # pdb.set_trace()
       
            
        return waveform, sample_rate, f0, uv, tgt_list_